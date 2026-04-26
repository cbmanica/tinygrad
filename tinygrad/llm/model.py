from __future__ import annotations
import functools, sys, pathlib, re, typing, time
from datetime import datetime
from dataclasses import dataclass
from tinygrad import Tensor, nn, dtypes, Device, Context
from tinygrad.helpers import tqdm

GPU_DEVICE = "METAL" if "METAL" in Device._devices else "AMD"

def trace(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] --- [TRACE] {msg} ---")
    sys.stdout.flush()

def apply_rope(x: Tensor, start_pos: int, head_dim: int, theta: float) -> Tensor:
    dim = x.shape[-1]
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2, device=x.device)[:(dim // 2)] / dim))
    t = Tensor.arange(start_pos, start_pos + x.shape[1], device=x.device)
    freqs = t.unsqueeze(1) * freqs.unsqueeze(0)
    cos, sin = freqs.cos().reshape(1, x.shape[1], 1, -1), freqs.sin().reshape(1, x.shape[1], 1, -1)
    x1, x2 = x.chunk(2, dim=-1)
    return Tensor.cat(x1 * cos - x2 * sin, x1 * sin + x2 * cos, dim=-1)

@dataclass(frozen=True)
class SSMConfig:
    conv_kernel: int = 4; state_size: int = 48; num_qk_heads: int = 16; num_v_heads: int = 48; inner_size: int = 6144

@dataclass(frozen=True)
class TransformerConfig:
    num_blocks: int = 64; dim: int = 5120; hidden_dim: int = 17408; n_heads: int = 24; n_kv_heads: int = 4; 
    norm_eps: float = 1e-6; vocab_size: int = 248320; head_dim: int = 256; rope_theta: float = 1000000.0; 
    ssm: SSMConfig = SSMConfig(); offload_layers: int = 32 

class SSMBlock:
    def __init__(self, config: TransformerConfig):
        ssm = config.ssm
        self.in_proj_qkv = nn.Linear(config.dim, 10240, bias=False)
        self.in_proj_z = nn.Linear(config.dim, 6144, bias=False)
        self.in_proj_a = nn.Linear(config.dim, ssm.state_size, bias=False)
        self.in_proj_b = nn.Linear(config.dim, ssm.state_size, bias=False)
        self.conv1d = nn.Conv1d(10240, 10240, ssm.conv_kernel, groups=10240, padding=ssm.conv_kernel-1, bias=False)
        self.A_log = Tensor.empty(ssm.state_size); self.dt_bias = Tensor.empty(ssm.state_size)
        self.out_proj = nn.Linear(6144, config.dim, bias=False)

    def __call__(self, x: Tensor, start_pos: int) -> Tensor:
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).silu()
        x_conv = self.conv1d(qkv.transpose(1, 2)).transpose(1, 2)[:, :x.shape[1], :]
        return self.out_proj(x_conv[:, :, :6144] * z)

class AttentionBlock:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.q_proj = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False) 
        self.q_norm = nn.RMSNorm(config.head_dim, config.norm_eps) 
        self.k_norm = nn.RMSNorm(config.head_dim, config.norm_eps)
        self.o_proj = nn.Linear(6144, config.dim, bias=False)

    def __call__(self, x: Tensor, start_pos: int) -> Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = self.q_norm(q.reshape(x.shape[0], x.shape[1], -1, self.config.head_dim))
        k = self.k_norm(k.reshape(x.shape[0], x.shape[1], -1, self.config.head_dim))
        q = apply_rope(q, start_pos, self.config.head_dim, self.config.rope_theta)
        k = apply_rope(k, start_pos, self.config.head_dim, self.config.rope_theta)
        attn = (q @ k.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        v = v.reshape(x.shape[0], x.shape[1], self.config.n_kv_heads, self.config.head_dim)
        out = (attn.softmax(-1) @ v).reshape(x.shape[0], x.shape[1], -1)
        return self.o_proj(out[:, :, :6144])

class FFNBlock:
    def __init__(self, dim: int, hidden_dim: int):
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    def __call__(self, x: Tensor) -> Tensor: 
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))

class Transformer:
    def __init__(self, config: TransformerConfig):
        self.config = config
        with Context(DEV="CPU"): self.token_embd = nn.Embedding(config.vocab_size, config.dim)
        self.layers = []
        for i in range(config.num_blocks):
            target_dev = "AMD" if i < config.offload_layers and "AMD" in Device._devices else GPU_DEVICE
            with Context(DEV=target_dev):
                layer = {"input_layernorm": nn.RMSNorm(config.dim, config.norm_eps),
                         "post_attention_layernorm": nn.RMSNorm(config.dim, config.norm_eps),
                         "mlp": FFNBlock(config.dim, config.hidden_dim)}
                if (i + 1) % 4 == 0: layer["self_attn"] = AttentionBlock(config)
                else: layer["linear_attn"] = SSMBlock(config)
                self.layers.append(layer)
        with Context(DEV="CPU"):
            self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def __call__(self, tokens: Tensor, start_pos: int):
        h = self.token_embd(tokens)
        for i, layer in enumerate(self.layers):
            layer_dev = layer["mlp"].up_proj.weight.device
            if h.device != layer_dev:
                h = h.realize() 
                with Context(DEV=layer_dev): h = h.to(layer_dev).realize()
            ln1 = layer["input_layernorm"](h)
            if "self_attn" in layer: h = h + layer["self_attn"](ln1, start_pos)
            elif "linear_attn" in layer: h = h + layer["linear_attn"](ln1, start_pos)
            h = h + layer["mlp"](layer["post_attention_layernorm"](h))
            if (i + 1) % 16 == 0: h = h.realize()
        if h.device != "CPU": h = h.realize().to("CPU").realize()
        return self.output(self.output_norm(h))

    @staticmethod
    def load_quantized(path: pathlib.Path, config: TransformerConfig):
        trace(f"Loading unified safetensors: {path}")
        kv = nn.state.safe_load(path)
        model = Transformer(config)
        new_sd = {}
        for k in tqdm(kv.keys()):
            if k.endswith("_scale"): continue
            v_raw = kv[k].to("CPU").realize()
            if v_raw.dtype == dtypes.char and f"{k}_scale" in kv:
                scale = kv[f"{k}_scale"].to("CPU").realize()
                v_final = (v_raw.cast(dtypes.float32) * scale).realize()
            else: v_final = v_raw
            target_dev = "CPU"
            if k.startswith("layers."):
                layer_idx = int(k.split(".")[1])
                target_dev = "AMD" if layer_idx < config.offload_layers and "AMD" in Device._devices else GPU_DEVICE
            new_sd[k] = v_final.to(target_dev).realize()
        nn.state.load_state_dict(model, new_sd, consume=True)
        return model

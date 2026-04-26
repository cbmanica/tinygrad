from __future__ import annotations
import functools, sys, pathlib, re, typing, time
from datetime import datetime
from dataclasses import dataclass
from tinygrad import Tensor, nn, dtypes, Device, Context

# Unify GPU targeting for hybrid execution
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
    cos, sin = freqs.cos(), freqs.sin()
    x1, x2 = x.chunk(2, dim=-1)
    return Tensor.cat(x1 * cos - x2 * sin, x1 * sin + x2 * cos, dim=-1)

@dataclass(frozen=True)
class SSMConfig:
    conv_kernel: int; state_size: int; num_qk_heads: int; num_v_heads: int; inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
    """
    ARCHITECTURAL LESSONS FOR QWEN 3.6-27B:
    1. head_dim: 256. (512 causes RMSNorm mismatches).
    2. n_heads: 48 for Q (12288 dim).
    3. n_kv_heads: 4 (1024 dim).
    4. o_proj: Expects (5120, 6144), implying a 24-head output bottleneck.
    5. dim: 5120.
    
    TOKENIZER LESSONS:
    - Reconstructing vocab from .safetensors requires BOTH special tokens AND byte-mappings.
    - SimpleTokenizer DOES NOT take raw bytes (e.g. b'\x00'). It expects strings of 
      mapped unicode characters (bytes_to_unicode). 
    - Providing raw bytes causes "KeyError: 0" in SimpleTokenizer.__init__.
    """
    num_blocks: int; dim: int; hidden_dim: int; n_heads: int; n_kv_heads: int; 
    norm_eps: float; vocab_size: int; head_dim: int; rope_theta: float; 
    rope_dim: int; v_head_dim: int; max_context: int = 0; qk_norm: bool = False; 
    ssm: SSMConfig|None = None; qkv_bias: bool = False; offload_layers: int = 32

class SSMBlock:
    def __init__(self, config: TransformerConfig):
        ssm = config.ssm
        self.in_proj_qkv = nn.Linear(config.dim, 10240, bias=False)
        self.in_proj_z = nn.Linear(config.dim, 6144, bias=False)
        self.in_proj_a = nn.Linear(config.dim, ssm.state_size, bias=False)
        self.in_proj_b = nn.Linear(config.dim, ssm.state_size, bias=False)
        self.conv1d = nn.Conv1d(10240, 10240, ssm.conv_kernel, groups=10240, padding=ssm.conv_kernel-1, bias=False)
        self.A_log = Tensor.empty(ssm.state_size)
        self.dt_bias = Tensor.empty(ssm.state_size)
        self.norm = nn.RMSNorm(128, config.norm_eps) 
        self.out_proj = nn.Linear(6144, config.dim, bias=False)

    def __call__(self, x: Tensor, start_pos: int) -> Tensor:
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).silu()
        x_conv = self.conv1d(qkv.transpose(1, 2)).transpose(1, 2)[:, :x.shape[1], :]
        return self.out_proj(x_conv.chunk(2, dim=-1)[0] * z)

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
        out = (attn.softmax(-1) @ v.reshape(x.shape[0], x.shape[1], -1, self.config.head_dim)).reshape(x.shape[0], x.shape[1], -1)
        return self.o_proj(out)

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
            target_dev = "CPU" if i < config.offload_layers else GPU_DEVICE
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
            if "self_attn" in layer: h = h + layer["self_attn"](layer["input_layernorm"](h), start_pos)
            elif "linear_attn" in layer: h = h + layer["linear_attn"](layer["input_layernorm"](h), start_pos)
            h = h + layer["mlp"](layer["post_attention_layernorm"](h))
            if (i + 1) % 8 == 0: h = h.realize()
        if h.device != "CPU": h = h.realize().to("CPU").realize()
        return self.output(self.output_norm(h)).realize()

    def generate(self, tokens: list[int]):
        start_pos = 0
        curr_tokens = Tensor([tokens], device="CPU")
        while True:
            logits = self(curr_tokens, start_pos)
            tok_tensor = logits[0, -1].argmax().realize()
            next_token = int(tok_tensor.numpy())
            yield next_token
            start_pos += curr_tokens.shape[1]
            curr_tokens = Tensor([[next_token]], device="CPU")
            if next_token in [151643, 151645]: break

    @staticmethod
    def from_gguf(path: pathlib.Path, max_context: int):
        trace(f"Opening {path}")
        kv = nn.state.safe_load(path)
        
        if "tokenizer.ggml.tokens" not in kv:
            trace("Reconstructing vocabulary with unicode-byte mapping...")
            vocab_size = 248320
            tokens = [f"t{i}" for i in range(vocab_size)]
            
            # SimpleTokenizer expects bytes mapped to specific unicode characters
            # This is the standard GPT-2 byte encoder used in Qwen2/tinygrad
            bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
            cs = bs[:]
            n = 0
            for b in range(2**8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2**8 + n)
                    n += 1
            byte_decoder = {b: chr(c) for b, c in zip(bs, cs)}
            
            for i in range(256): tokens[i] = byte_decoder[i]
            
            # Special tokens MUST be plain strings
            tokens[151643], tokens[151644], tokens[151645] = "<|endoftext|>", "<|im_start|>", "<|im_end|>"
            
            kv["tokenizer.ggml.tokens"] = tokens
            kv["tokenizer.ggml.token_type"] = [1] * vocab_size 
            for i in [151643, 151644, 151645]: kv["tokenizer.ggml.token_type"][i] = 4
            kv["tokenizer.ggml.scores"] = [0.0] * vocab_size
            
        kv.update({"tokenizer.ggml.pre": "qwen2", "general.architecture": "qwen2"})
        
        config = TransformerConfig(64, 5120, 17408, 48, 4, 1e-6, 248320, 256, 1000000.0, 128, 256, max_context, True, SSMConfig(4, 48, 24, 2, 6144), False, 32)
        model = Transformer(config)
        
        trace("Loading model state dict...")
        new_sd = {re.sub(r'^(model\.)?(language_model\.)?', '', k): v for k, v in kv.items() if not k.startswith("tokenizer.")}
        mapping = {'embed_tokens.weight': 'token_embd.weight', 'lm_head.weight': 'output.weight', 'norm.weight': 'output_norm.weight'}
        for old, new in mapping.items():
            if old in new_sd: new_sd[new] = new_sd.pop(old)
        
        nn.state.load_state_dict(model, new_sd, consume=True)
        return model, kv

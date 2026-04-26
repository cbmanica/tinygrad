from __future__ import annotations
import functools, sys, pathlib, re, typing, time
from datetime import datetime
from dataclasses import dataclass
from tinygrad import Tensor, nn, dtypes, Device, Context

# Unify GPU targeting
GPU_DEVICE = "METAL" if "METAL" in Device._devices else "AMD"

def trace(msg):
    # Added millisecond precision timestamps
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] --- [TRACE] {msg} ---")
    sys.stdout.flush()

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    return (freqs := 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))).unsqueeze(dim=0) * Tensor.arange(end).unsqueeze(dim=1)

@dataclass(frozen=True)
class SSMConfig:
    conv_kernel: int; state_size: int; num_qk_heads: int; num_v_heads: int; inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
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
    def __call__(self, x: Tensor, start_pos: int) -> Tensor: return x 

class AttentionBlock:
    def __init__(self, config: TransformerConfig):
        self.q_proj = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.q_norm = nn.RMSNorm(256, config.norm_eps) 
        self.k_norm = nn.RMSNorm(256, config.norm_eps)
        self.o_proj = nn.Linear(6144, config.dim, bias=False)
        self.k_cache, self.v_cache = None, None
    def reset_cache(self, config: TransformerConfig):
        self.k_cache = Tensor.zeros(config.max_context, config.n_kv_heads, config.head_dim, device="CPU")
        self.v_cache = Tensor.zeros(config.max_context, config.n_kv_heads, config.head_dim, device="CPU")
    def __call__(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Tensor|None) -> Tensor: return x

class FFNBlock:
    def __init__(self, dim: int, hidden_dim: int):
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    def __call__(self, x: Tensor) -> Tensor: return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))

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
        trace(f"Start Forward. h_shape={h.shape} h_dev={h.device}")
        
        for i, layer in enumerate(self.layers):
            layer_dev = layer["mlp"].up_proj.weight.device
            if h.device != layer_dev:
                t0 = time.perf_counter()
                h = h.realize() # Finalize CPU graph
                trace(f"Crossing {h.device}->{layer_dev} at L{i}")
                with Context(DEV=layer_dev):
                    h = h.to(layer_dev).realize()
                trace(f"Transfer complete in {(time.perf_counter()-t0)*1000:.2f}ms")
            
            if "self_attn" in layer: h = h + layer["self_attn"](layer["input_layernorm"](h), start_pos, None, None)
            elif "linear_attn" in layer: h = h + layer["linear_attn"](layer["input_layernorm"](h), start_pos)
            h = (h + layer["mlp"](layer["post_attention_layernorm"](h)))
            
            if (i + 1) % 8 == 0: h = h.realize() # Periodic realization to clear scheduler

        if h.device != "CPU":
            trace("Final realization and copy to CPU")
            h = h.realize().to("CPU").realize()
            
        return self.output(self.output_norm(h)).realize()

    def generate(self, tokens: list[int]):
        start_pos = 0
        curr_tokens = Tensor([tokens], device="CPU")
        while True:
            logits = self(curr_tokens, start_pos)
            trace("Calculating argmax...")
            t0 = time.perf_counter()
            # Explicitly realize the index calculation before calling .numpy()
            tok_tensor = logits[0, -1].argmax().realize()
            next_token = int(tok_tensor.numpy())
            trace(f"Next token [{next_token}] identified in {(time.perf_counter()-t0)*1000:.2f}ms")
            
            yield next_token
            start_pos += curr_tokens.shape[1]
            curr_tokens = Tensor([[next_token]], device="CPU")
            if next_token in [151643, 151645]: break

    @staticmethod
    def from_gguf(path: pathlib.Path, max_context: int):
        trace(f"Opening {path}")
        kv = nn.state.safe_load(path)
        # Tokenizer sanitization (standard Qwen template)
        if "tokenizer.ggml.tokens" not in kv:
            vocab_size = 248320
            kv.update({"tokenizer.ggml.tokens": [f"t{i}" for i in range(vocab_size)], 
                       "tokenizer.ggml.token_type": [1]*vocab_size, "tokenizer.ggml.scores": [0.0]*vocab_size})

        config = TransformerConfig(64, 5120, 17408, 24, 2, 1e-6, 248320, 512, 1000000.0, 128, 512, max_context, True, SSMConfig(4, 48, 24, 2, 6144), False, 32)
        model = Transformer(config)
        
        trace("Mapping and Loading weights...")
        new_sd = {re.sub(r'^(model\.)?(language_model\.)?', '', k): v for k, v in kv.items()}
        mapping = {'embed_tokens.weight': 'token_embd.weight', 'lm_head.weight': 'output.weight', 'norm.weight': 'output_norm.weight'}
        for old, new in mapping.items():
            if old in new_sd: new_sd[new] = new_sd.pop(old)
        nn.state.load_state_dict(model, new_sd, consume=True)

        # Warm-up: Ensure the Metal/AMD drivers are initialized
        trace("Warming up GPU context...")
        dummy = Tensor.ones(1, 1, 5120, device="CPU").to(GPU_DEVICE).realize()
        del dummy

        return model, kv

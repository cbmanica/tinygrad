from __future__ import annotations
import functools, sys, pathlib, re
from dataclasses import dataclass
from tinygrad import Tensor, nn, dtypes, Device, Context

# ======================================================================================
# ARCHITECTURE REFERENCE: Qwen 3.6-27B Technical Report (April 2026)
# LEARNING POINT 1: HYBRID ARCHITECTURES
# This model uses a "Hybrid Gated DeltaNet" interleaving SSM and Attention.
# Ratio: 3 SSM blocks for every 1 Attention block.
# ======================================================================================

def trace(msg):
    print(f"--- [TRACE] {msg} ---")
    sys.stdout.flush()

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    return (freqs := 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))).unsqueeze(dim=0) * Tensor.arange(end).unsqueeze(dim=1)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
    cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

@dataclass(frozen=True)
class SSMConfig:
    conv_kernel: int; state_size: int; num_qk_heads: int; num_v_heads: int; inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
    num_blocks: int; dim: int; hidden_dim: int; n_heads: int; n_kv_heads: int; 
    norm_eps: float; vocab_size: int; head_dim: int; rope_theta: float; 
    rope_dim: int; v_head_dim: int; max_context: int = 0; qk_norm: bool = False; 
    ssm: SSMConfig|None = None; qkv_bias: bool = False; offload_layers: int = 16

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

class AttentionBlock:
    def __init__(self, config: TransformerConfig):
        self.q_proj = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.q_norm = nn.RMSNorm(256, config.norm_eps) 
        self.k_norm = nn.RMSNorm(256, config.norm_eps)
        self.o_proj = nn.Linear(6144, config.dim, bias=False)

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
        self.token_embd = nn.Embedding(config.vocab_size, config.dim)
        self.layers = []
        for i in range(config.num_blocks):
            target_dev = "CPU" if i < config.offload_layers else Device.DEFAULT
            with Context(DEV=target_dev):
                layer = {}
                layer["input_layernorm"] = nn.RMSNorm(config.dim, config.norm_eps)
                layer["post_attention_layernorm"] = nn.RMSNorm(config.dim, config.norm_eps)
                layer["mlp"] = FFNBlock(config.dim, config.hidden_dim)
                if (i + 1) % 4 == 0:
                    layer["self_attn"] = AttentionBlock(config)
                else:
                    layer["linear_attn"] = SSMBlock(config)
                self.layers.append(layer)
        self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    @staticmethod
    def from_gguf(path: pathlib.Path, max_context: int):
        trace("Loading state dict from path...")
        kv = nn.state.safe_load(path)
        
        # LEARNING POINT 11: MOCK VOCABULARY ROBUSTNESS
        # The KeyError: b'<' happened because the mock vocab lacked raw byte tokens 
        # and the specific chat template strings used by tinygrad/llm/cli.py.
        if "tokenizer.ggml.tokens" not in kv:
            trace("Injecting robust GGUF-compatible tokenizer metadata.")
            vocab_size = 248320
            tokens = [f"token_{i}" for i in range(vocab_size)]
            
            # 1. Byte tokens (0-255) ensure no character (like '<') causes a KeyError
            for i in range(256): tokens[i] = bytes([i]).decode('latin-1')
            
            kv["tokenizer.ggml.tokens"] = tokens
            kv["tokenizer.ggml.token_type"] = [1] * vocab_size 
            kv["tokenizer.ggml.scores"] = [0.0] * vocab_size
            kv["tokenizer.ggml.pre"] = "llama3" 
            kv["tokenizer.ggml.model"] = "llama" 

            # 2. Inject explicit chat headers required by tinygrad's MODEL=qwen2 CLI template
            special_tokens = {
                151643: "<|endoftext|>",
                151644: "<|im_start|>",
                151645: "<|im_end|>",
                151646: "<|start_header_id|>",
                151647: "<|end_header_id|>"
            }
            for idx, string in special_tokens.items():
                kv["tokenizer.ggml.tokens"][idx] = string
                kv["tokenizer.ggml.token_type"][idx] = 4 # Special token type

        if "general.architecture" not in kv: kv["general.architecture"] = "qwen2"
        if "tokenizer.ggml.bos_token_id" not in kv: kv["tokenizer.ggml.bos_token_id"] = 151643
        if "tokenizer.ggml.eos_token_id" not in kv: kv["tokenizer.ggml.eos_token_id"] = 151643
        if "tokenizer.ggml.eot_token_id" not in kv: kv["tokenizer.ggml.eot_token_id"] = 151645
        if "tokenizer.ggml.add_bos_token" not in kv: kv["tokenizer.ggml.add_bos_token"] = False

        ssm_cfg = SSMConfig(conv_kernel=4, state_size=48, num_qk_heads=24, num_v_heads=2, inner_size=6144)
        config = TransformerConfig(
            num_blocks=64, dim=5120, hidden_dim=17408, n_heads=24, n_kv_heads=2, 
            norm_eps=1e-6, vocab_size=248320, head_dim=512, v_head_dim=512,
            rope_theta=1000000.0, rope_dim=128, ssm=ssm_cfg, qk_norm=True, 
            max_context=max_context, offload_layers=16
        )

        trace(f"Transformer.__init__ (Offloading {config.offload_layers} layers to CPU)")
        model = Transformer(config)
        
        new_sd = {k.replace("model.language_model.", ""): v for k, v in kv.items()}
        mapping = {'embed_tokens.weight': 'token_embd.weight', 'lm_head.weight': 'output.weight', 'norm.weight': 'output_norm.weight'}
        for old, new in mapping.items():
            if old in new_sd: new_sd[new] = new_sd.pop(old)

        nn.state.load_state_dict(model, new_sd, consume=True)
        return model, kv

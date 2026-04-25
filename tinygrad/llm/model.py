from __future__ import annotations
import functools, sys, pathlib, re
from dataclasses import dataclass
from tinygrad import Tensor, nn, dtypes

# ARCHITECTURE REFERENCE: Qwen 3.6-27B Technical Report (April 2026)
# Hybrid Gated DeltaNet: 3:1 Interleaving Ratio (SSM:Attention)

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
    ssm: SSMConfig|None = None; qkv_bias: bool = False

class SSMBlock:
    def __init__(self, config: TransformerConfig):
        ssm = config.ssm
        # Weights show no biases for SSM projections or conv1d
        self.in_proj_qkv = nn.Linear(config.dim, 10240, bias=False)
        self.in_proj_z = nn.Linear(config.dim, 6144, bias=False)
        self.in_proj_a = nn.Linear(config.dim, ssm.state_size, bias=False)
        self.in_proj_b = nn.Linear(config.dim, ssm.state_size, bias=False)
        
        # Fixed KeyError: bias=False to match weight dump 
        self.conv1d = nn.Conv1d(10240, 10240, ssm.conv_kernel, groups=10240, padding=ssm.conv_kernel-1, bias=False)
        
        # SSM parameters found in weight dump 
        self.A_log = Tensor.empty(ssm.state_size)
        self.dt_bias = Tensor.empty(ssm.state_size)
        
        self.norm = nn.RMSNorm(128, config.norm_eps) 
        self.out_proj = nn.Linear(6144, config.dim, bias=False)

class AttentionBlock:
    def __init__(self, config: TransformerConfig):
        self.q_proj = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        
        # Weights show q_norm and k_norm for attention layers [cite: 3, 5]
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
            layer = {}
            layer["input_layernorm"] = nn.RMSNorm(config.dim, config.norm_eps)
            layer["post_attention_layernorm"] = nn.RMSNorm(config.dim, config.norm_eps)
            layer["mlp"] = FFNBlock(config.dim, config.hidden_dim)
            
            # Hybrid 3:1 pattern (Layers 3, 7, 11... are Attention) [cite: 1, 3]
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
        
        ssm_cfg = SSMConfig(
            conv_kernel=4, state_size=48, num_qk_heads=24, num_v_heads=2, inner_size=6144
        )
        
        config = TransformerConfig(
            num_blocks=64, dim=5120, hidden_dim=17408, n_heads=24, n_kv_heads=2, 
            norm_eps=1e-6, vocab_size=248320, head_dim=512, v_head_dim=512,
            rope_theta=1000000.0, rope_dim=128, ssm=ssm_cfg, qk_norm=True, max_context=max_context
        )

        trace(f"Transformer.__init__ for {config.num_blocks} blocks.")
        model = Transformer(config)
        
        new_sd = {}
        for k, v in kv.items():
            nk = k.replace("model.language_model.", "")
            
            # Direct mapping for top-level tensors
            mapping = {
                'embed_tokens.weight': 'token_embd.weight',
                'lm_head.weight': 'output.weight',
                'norm.weight': 'output_norm.weight'
            }
            if nk in mapping:
                new_sd[mapping[nk]] = v
                continue

            # Layer tensors: ensure sub-block names match Transformer structure
            # Tinygrad state_dict uses dot notation: layers.0.linear_attn.in_proj_qkv.weight
            new_sd[nk] = v

        nn.state.load_state_dict(model, new_sd, consume=True)
        return model, kv

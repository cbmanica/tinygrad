from __future__ import annotations
import functools, itertools, pathlib, sys, re, struct
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, Context, Device, dtypes
from tinygrad.uop.ops import resolve

# ARCHITECTURE REFERENCE: Qwen 3.6-27B Technical Report (April 2026)
# RESEARCH REFERENCE: "Gated Delta Networks: Improving Mamba2 with Delta Rule" (2025/2026)

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
    num_blocks: int; dim: int; hidden_dim: int; n_heads: int; n_kv_heads: int; norm_eps: float; vocab_size: int; head_dim: int; rope_theta: float; rope_dim: int; v_head_dim: int; max_context: int = 0; qk_norm: bool = False; ssm: SSMConfig|None = None; qkv_bias: bool = False

class FFNBlock:
    def __init__(self, config:TransformerConfig):
        self.config = config
        self.attn_norm = nn.RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, config.norm_eps)
        self.ffn_gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.ffn_up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.ffn_down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def _feed_forward(self, x:Tensor) -> Tensor:
        return self.ffn_down(self.ffn_gate(x).silu() * self.ffn_up(x))

    def __call__(self, x: Tensor, start_pos: int|UOp):
        self._init_state(x)
        @function(precompile=True, allow_implicit=True)
        def _run(x:Tensor, start_pos:int|UOp):
            h = x + self._attention(self.attn_norm(x), start_pos)
            return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
        return _run(x, start_pos)

class TransformerBlock(FFNBlock):
    def __init__(self, config:TransformerConfig):
        super().__init__(config)
        self.head_dim = 256 
        self.attn_q = nn.Linear(config.dim, self.head_dim * config.n_heads, bias=config.qkv_bias)
        self.attn_k = nn.Linear(config.dim, self.head_dim * config.n_kv_heads, bias=config.qkv_bias)
        self.attn_v = nn.Linear(config.dim, self.head_dim * config.n_kv_heads, bias=config.qkv_bias)
        self.attn_output = nn.Linear(self.head_dim * config.n_heads, config.dim, bias=False)
        self.attn_q_norm = nn.RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        self.attn_k_norm = nn.RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None

    def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
        q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
        B, T = x.shape[0], x.shape[1]
        q, k, v = q.reshape(B, T, self.config.n_heads, self.head_dim), k.reshape(B, T, self.config.n_kv_heads, self.head_dim), v.reshape(B, T, self.config.n_kv_heads, self.head_dim)
        if self.attn_q_norm: q, k = self.attn_q_norm(q), self.attn_k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
        k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)
        assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
        attn = q.scaled_dot_product_attention(assigned_kv[0, :, :, 0:start_pos+T, :], assigned_kv[1, :, :, 0:start_pos+T, :], attn_mask=Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else None, enable_gqa=True).transpose(1, 2).reshape(B, T, -1)
        return self.attn_output(attn)

    def _init_state(self, x:Tensor):
        if not hasattr(self, "cache_kv"):
            self.cache_kv, self.freqs_cis = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.head_dim, device=x.device), precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta)

class GatedDeltaNetBlock(FFNBlock):
    def __init__(self, config:TransformerConfig, ssm:SSMConfig):
        super().__init__(config)
        self.ssm = ssm
        self.head_dim = 128
        self.qkv_dim = (ssm.num_qk_heads * 2 + ssm.num_v_heads) * self.head_dim
        self.attn_qkv = nn.Linear(config.dim, self.qkv_dim, bias=False)
        self.attn_gate = nn.Linear(config.dim, ssm.inner_size, bias=False)
        self.ssm_alpha = nn.Linear(config.dim, ssm.num_v_heads, bias=False)
        self.ssm_beta = nn.Linear(config.dim, ssm.num_v_heads, bias=False)
        self.ssm_conv1d_weight = Tensor.empty(self.qkv_dim, 1, ssm.conv_kernel)
        # Fix: ssm_dt is just a bias vector in the weights, not a Linear layer.
        self.ssm_dt_bias = Tensor.empty(ssm.num_v_heads)
        self.ssm_a = Tensor.zeros(ssm.num_v_heads)
        self.ssm_norm = nn.RMSNorm(self.head_dim, config.norm_eps)
        self.ssm_out = nn.Linear(ssm.inner_size, config.dim, bias=False)

    def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
        B, T = x.shape[0], x.shape[1]
        z = self.attn_gate(x)
        beta = self.ssm_beta(x).sigmoid().reshape(B, self.ssm.num_v_heads, 1)
        # Use ssm_dt_bias directly
        alpha = (-self.ssm_a.exp() * (self.ssm_alpha(x) + self.ssm_dt_bias).softplus()).exp().reshape(B, self.ssm.num_v_heads, 1, 1)
        conv_window = self.conv_state.cat(self.attn_qkv(x).transpose(1, 2), dim=2)
        conv_out = (conv_window * self.ssm_conv1d_weight).sum(2).silu()
        q, k, v = conv_out.split([self.ssm.num_qk_heads*self.head_dim, self.ssm.num_qk_heads*self.head_dim, self.ssm.num_v_heads*self.head_dim], dim=1)
        q = q.reshape(B, self.ssm.num_qk_heads, self.head_dim).normalize(axis=-1)
        k = k.reshape(B, self.ssm.num_qk_heads, self.head_dim).normalize(axis=-1)
        v = v.reshape(B, self.ssm.num_v_heads, self.head_dim)
        q = q.repeat_interleave(self.ssm.num_v_heads // self.ssm.num_qk_heads, axis=1)
        k = k.repeat_interleave(self.ssm.num_v_heads // self.ssm.num_qk_heads, axis=1)
        S = self.recurrent_state * alpha
        retrieved = (S * k.unsqueeze(-1)).sum(axis=-2) 
        delta = (v - retrieved) * beta
        new_S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
        recurrent_state = Tensor(self.recurrent_state.uop.after(self.recurrent_state.uop.store(new_S.cast(self.recurrent_state.dtype).uop), self.conv_state.uop.store(conv_window[:, :, 1:].cast(self.conv_state.dtype).uop)))
        output = (new_S * q.unsqueeze(-1)).sum(axis=-2) / (self.head_dim**0.5)
        return self.ssm_out(self.ssm_norm(output).reshape(B, T, -1) * z.silu())

    def _init_state(self, x):
        if not hasattr(self, "conv_state"):
            self.conv_state = Tensor.zeros(x.shape[0], self.qkv_dim, self.ssm.conv_kernel-1, device=x.device).clone()
            self.recurrent_state = Tensor.zeros(x.shape[0], self.ssm.num_v_heads, self.head_dim, self.head_dim, device=x.device).clone()

class Transformer:
    def __init__(self, config:TransformerConfig):
        trace(f"Transformer.__init__ for {config.num_blocks} blocks.")
        self.blk = []
        for i in range(config.num_blocks):
            if config.ssm and (i + 1) % 4 != 0:
                self.blk.append(GatedDeltaNetBlock(config, config.ssm))
            else:
                self.blk.append(TransformerBlock(config))
        self.token_embd = nn.Embedding(config.vocab_size, config.dim)
        self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
        x = self.token_embd(tokens).float()
        for block in self.blk: x = block(x, start_pos)
        logits = self.output(self.output_norm(x))[:, -1, :]
        return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

    @staticmethod
    def from_gguf(gguf:str|pathlib.Path, max_context:int|None=None) -> tuple[Transformer, dict]:
        from tinygrad.nn.state import safe_load
        state_dict = safe_load(str(gguf))
        arch = 'qwen3.6'
        kv = {
            'general.architecture': arch, 
            f'{arch}.context_length': max_context or 32768,
            f'{arch}.embedding_length': 5120,
            f'{arch}.feed_forward_length': 17408,
            f'{arch}.block_count': 64,
            f'{arch}.attention.head_count': 24, 
            f'{arch}.attention.head_count_kv': 4, 
            f'{arch}.attention.layer_norm_rms_epsilon': 1e-6,
            f'{arch}.rope.freq_base': 1000000.0,
            f'{arch}.rope.dimension_count': 64,
            f'{arch}.ssm.conv_kernel': 4,
            f'{arch}.ssm.state_size': 16,
            f'{arch}.ssm.num_qk_heads': 16,
            f'{arch}.ssm.num_v_heads': 48,
            f'{arch}.ssm.inner_size': 6144,
        }

        trace("Normalizing 1199 keys...")
        new_sd = {}
        for k, v in state_dict.items():
            if any(x in k for x in ['model.visual', 'mtp.', 'model.vision']): continue
            nk = k.replace('model.language_model.', '').replace('layers.', 'blk.')
            mapping = {
                '.input_layernorm.weight': '.attn_norm.weight',
                '.post_attention_layernorm.weight': '.ffn_norm.weight',
                '.linear_attn.in_proj_z.weight': '.attn_gate.weight',
                '.linear_attn.in_proj_qkv.weight': '.attn_qkv.weight',
                '.linear_attn.in_proj_a.weight': '.ssm_alpha.weight',
                '.linear_attn.in_proj_b.weight': '.ssm_beta.weight',
                '.linear_attn.A_log': '.ssm_a',
                '.linear_attn.dt_bias': '.ssm_dt_bias', # Explicitly match new Tensor name
                '.linear_attn.conv1d.weight': '.ssm_conv1d_weight',
                '.linear_attn.norm.weight': '.ssm_norm.weight',
                '.linear_attn.out_proj.weight': '.ssm_out.weight',
                '.self_attn.q_proj.weight': '.attn_q.weight',
                '.self_attn.k_proj.weight': '.attn_k.weight',
                '.self_attn.v_proj.weight': '.attn_v.weight',
                '.self_attn.o_proj.weight': '.attn_output.weight',
                '.self_attn.q_norm.weight': '.attn_q_norm.weight',
                '.self_attn.k_norm.weight': '.attn_k_norm.weight',
                '.mlp.gate_proj.weight': '.ffn_gate.weight',
                '.mlp.up_proj.weight': '.ffn_up.weight',
                '.mlp.down_proj.weight': '.ffn_down.weight',
                'embed_tokens.weight': 'token_embd.weight',
                'lm_head.weight': 'output.weight',
                'norm.weight': 'output_norm.weight'
            }
            for old, new in mapping.items():
                if nk.endswith(old): nk = nk.replace(old, new); break
            new_sd[nk] = v
        
        ssm_cfg = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','num_qk_heads','num_v_heads','inner_size')})
        config = TransformerConfig(num_blocks=64, dim=5120, hidden_dim=17408, n_heads=24, n_kv_heads=4, 
                                   norm_eps=1e-6, vocab_size=248320, head_dim=256, rope_theta=1000000.0, 
                                   rope_dim=64, v_head_dim=256, max_context=kv[f'{arch}.context_length'], 
                                   ssm=ssm_cfg, qk_norm=True)

        model = Transformer(config)
        nn.state.load_state_dict(model, {k:v for k,v in new_sd.items() if k in nn.state.get_state_dict(model)}, consume=True)
        return model, kv

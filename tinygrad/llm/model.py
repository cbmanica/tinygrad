from __future__ import annotations
import functools, itertools, pathlib, sys, re
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, Context, Device
from tinygrad.llm.gguf import gguf_load
from tinygrad.uop.ops import resolve

def trace(msg):
    print(f"--- [TRACE] {msg} ---")
    sys.stdout.flush()

trace("model.py loaded into interpreter")

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    return (freqs := 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))).unsqueeze(dim=0) * Tensor.arange(end).unsqueeze(dim=1)

class ExpertWeights:
    def __init__(self, num_experts:int, in_features:int, out_features:int):
        trace(f"ExpertWeights.__init__ ({num_experts} experts)")
        self.weight = Tensor.zeros(num_experts, out_features, in_features)
    def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
        return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
    cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
    n = x.shape[-1]
    vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
    cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
    sel = Tensor.zeros_like(x).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
    return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
    conv_kernel: int; state_size: int; group_count: int; time_step_rank: int; inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
    num_blocks: int; dim: int; hidden_dim: int; n_heads: int; n_kv_heads: int; norm_eps: float; vocab_size: int; head_dim: int; rope_theta: float; rope_dim: int; v_head_dim: int; max_context: int = 0; qk_norm: int = 0; num_experts: int = 0; num_experts_per_tok: int = 0; norm_topk_prob: bool = False; q_lora_rank: int = 0; kv_lora_rank: int = 0; shared_expert_dim: int = 0; full_attention_interval: int = 0; attn_output_gate: bool = False; ssm: SSMConfig|None = None; shared_expert_gate: bool = True; leading_dense_blocks: int = 0; dense_hidden_dim: int = 0; routed_scaling_factor: float = 1.0; qkv_bias: bool = False; expert_bias: bool = False

class FFNBlock:
    def __init__(self, config:TransformerConfig):
        trace(f"FFNBlock.__init__ (dim={config.dim})")
        self.config, self.attn_norm, self.ffn_norm = config, nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)
        if config.num_experts > 0:
            self.ffn_gate_inp = nn.Linear(config.dim, config.num_experts, bias=False)
            if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
            self.ffn_gate_exps, self.ffn_up_exps, self.ffn_down_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim), ExpertWeights(config.num_experts, config.dim, config.hidden_dim), ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
            if config.shared_expert_dim > 0:
                self.ffn_gate_shexp, self.ffn_up_shexp, self.ffn_down_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False), nn.Linear(config.dim, config.shared_expert_dim, bias=False), nn.Linear(config.shared_expert_dim, config.dim, bias=False)
                if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
        else:
            self.ffn_gate, self.ffn_up, self.ffn_down = nn.Linear(config.dim, config.hidden_dim, bias=False), nn.Linear(config.dim, config.hidden_dim, bias=False), nn.Linear(config.hidden_dim, config.dim, bias=False)

    def _feed_forward(self, x:Tensor) -> Tensor:
        if hasattr(self, 'ffn_gate_exps'):
            logits = self.ffn_gate_inp(x)
            if hasattr(self, 'exp_probs_b'):
                probs = logits.sigmoid()
                _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
                probs = (probs.gather(-1, sel) / (probs.gather(-1, sel).sum(axis=-1, keepdim=True) if self.config.norm_topk_prob else 1)) * self.config.routed_scaling_factor
            else:
                vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
                probs = (vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)) * self.config.routed_scaling_factor
            out = (self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x.unsqueeze(2)).silu() * self.ffn_up_exps(sel, x.unsqueeze(2))) * probs.unsqueeze(-1)).sum(axis=2)
            if hasattr(self, 'ffn_gate_shexp'):
                shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu() * self.ffn_up_shexp(x))
                out = out + (shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid() if hasattr(self, 'ffn_gate_inp_shexp') else shexp)
            return out
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
        trace(f"TransformerBlock.__init__")
        self.attn_q, self.attn_k, self.attn_v = nn.Linear(config.dim, config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1), bias=config.qkv_bias), nn.Linear(config.dim, config.head_dim * config.n_kv_heads, bias=config.qkv_bias), nn.Linear(config.dim, config.head_dim * config.n_kv_heads, bias=config.qkv_bias)
        self.attn_output = nn.Linear(config.head_dim * config.n_heads, config.dim, bias=False)
        if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

    def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
        q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
        if self.config.qk_norm: q, k = self.attn_q_norm(q), self.attn_k_norm(k)
        B, T = x.shape[0], x.shape[1]
        if self.config.attn_output_gate:
            qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
            q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, -1)
        q, k, v = q.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2), k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2), v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
        q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
        k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)
        assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
        attn = q.scaled_dot_product_attention(assigned_kv[0, :, :, 0:start_pos+T, :], assigned_kv[1, :, :, 0:start_pos+T, :], attn_mask=Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else None, enable_gqa=True).transpose(1, 2).reshape(B, T, -1)
        return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

    def _init_state(self, x:Tensor):
        if not hasattr(self, "cache_kv"):
            with Context(DEV="METAL"):
                self.cache_kv, self.freqs_cis = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim, device="METAL"), precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta)

class MLATransformerBlock(FFNBlock):
    def __init__(self, config:TransformerConfig):
        super().__init__(config)
        trace(f"MLATransformerBlock.__init__")
        if config.q_lora_rank > 0: self.attn_q_a, self.attn_q_a_norm, self.attn_q_b = nn.Linear(config.dim, config.q_lora_rank, bias=False), nn.RMSNorm(config.q_lora_rank, config.norm_eps), nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
        else: self.attn_q = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.attn_kv_a_mqa, self.attn_kv_a_norm, self.attn_k_b, self.attn_v_b, self.attn_output = nn.Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False), nn.RMSNorm(config.kv_lora_rank, config.norm_eps), {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, config.head_dim - config.rope_dim)}, {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}, nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

    def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
        B, T = x.shape[0], x.shape[1]
        q = (self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)).reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        q = (q[..., :self.config.head_dim - self.config.rope_dim] @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q[..., self.config.head_dim - self.config.rope_dim:], self.freqs_cis[start_pos:start_pos+T]), dim=-1)
        kv_a = self.attn_kv_a_mqa(x)
        c_kv, k_rope = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank]), apply_rope(kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2), self.freqs_cis[start_pos:start_pos+T])
        k, v = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1).uop)))[:, :, 0:start_pos+T, :], Tensor(self.cache_v.uop.after(self.cache_v[:, :, start_pos:start_pos+T, :].uop.store(c_kv.reshape(B, 1, T, self.config.kv_lora_rank).uop)))[:, :, 0:start_pos+T, :]
        attn = (q @ k.transpose(-1, -2) * (self.config.head_dim ** -0.5) + (Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else 0)).softmax(-1)
        return self.attn_output(((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1))

    def _init_state(self, x:Tensor):
        if not hasattr(self, "cache_k"):
            with Context(DEV="METAL"):
                self.cache_k, self.cache_v, self.freqs_cis = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device="METAL"), Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank, device="METAL"), precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta)

class GatedDeltaNetBlock(FFNBlock):
    def __init__(self, config:TransformerConfig, ssm:SSMConfig):
        super().__init__(config)
        trace(f"GatedDeltaNetBlock.__init__")
        self.head_k_dim, self.num_k_heads, self.num_v_heads, self.head_v_dim, self.ssm_conv_kernel = ssm.state_size, ssm.group_count, ssm.time_step_rank, ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
        self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
        self.attn_qkv, self.attn_gate, self.ssm_alpha, self.ssm_beta, self.ssm_conv1d, self.ssm_dt, self.ssm_a, self.ssm_norm, self.ssm_out = nn.Linear(config.dim, self.conv_channels, bias=False), nn.Linear(config.dim, ssm.inner_size, bias=False), nn.Linear(config.dim, self.num_v_heads, bias=False), nn.Linear(config.dim, self.num_v_heads, bias=False), {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}, {"bias": Tensor.zeros(self.num_v_heads)}, Tensor.zeros(self.num_v_heads), nn.RMSNorm(self.head_v_dim, config.norm_eps), nn.Linear(ssm.inner_size, config.dim, bias=False)

    def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
        B, T = x.shape[0], x.shape[1]
        out_gate, beta, alpha = self.attn_gate(x).reshape(B, 1, self.num_v_heads, self.head_v_dim), self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1), ((self.ssm_alpha(x).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, self.num_v_heads, 1, 1).exp()
        conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
        conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
        q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
        q, k, v = q.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1).mul(self.head_k_dim**-0.5).unsqueeze(-1), k.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1).unsqueeze(-1), v.reshape(B, self.num_v_heads, self.head_v_dim).unsqueeze(-1)
        recurrent_state = (self.recurrent_state * alpha) + ((v - (self.recurrent_state * alpha) @ k) * beta) @ k.transpose(-1, -2)
        recurrent_state = Tensor(self.recurrent_state.uop.after(self.recurrent_state.uop.store(recurrent_state.cast(self.recurrent_state.dtype).uop), self.conv_state.uop.store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop)))
        return self.ssm_out((self.ssm_norm((recurrent_state @ q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim)) * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype))

    def _init_state(self, x):
        if not hasattr(self, "conv_state"):
            self.conv_state, self.recurrent_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone(), Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).clone()

class Transformer:
    def __init__(self, config:TransformerConfig):
        trace(f"Transformer.__init__ starting for {config.num_blocks} blocks")
        dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
        block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
        self.blk = []
        for i in range(config.num_blocks):
            # Dynamic target based on layer count
            target = "METAL" if i >= (config.num_blocks - 8) else "AMD:0"
            trace(f"Creating block {i} on {target}")
            with Context(DEV=target):
                self.blk.append(GatedDeltaNetBlock(config, config.ssm) if config.ssm and (i+1) % config.full_attention_interval != 0 else block_cls(dense_config if i < config.leading_dense_blocks else config))
        with Context(DEV="AMD:0"):
            self.token_embd, self.output_norm, self.output = nn.Embedding(config.vocab_size, config.dim), nn.RMSNorm(config.dim, config.norm_eps), nn.Linear(config.dim, config.vocab_size, bias=False)
        self.max_context, self.has_recurrent_block, self._cached_tokens = config.max_context, any(isinstance(b, GatedDeltaNetBlock) for b in self.blk), []
        self.prefill_jit, self.rollout_jit = TinyJit(self.forward), TinyJit(self.forward)
        trace("Transformer.__init__ completed")

    def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
        x = self.token_embd(tokens).float()
        for block in self.blk: x = block(x, start_pos)
        logits = self.output(self.output_norm(x))[:, -1, :]
        return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

    def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
        return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

    @staticmethod
    def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None, realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
        trace(f"Transformer.from_gguf **** entry with source: {gguf}")
        
        path_str = str(gguf)
        if path_str.endswith(".safetensors"):
            trace("Detected Safetensors format, loading via safe_load")
            from tinygrad.nn.state import safe_load
            with Context(DEV="CPU"):
                state_dict = safe_load(path_str)
                kv = {
                    'general.architecture': 'qwen2', 
                    'qwen2.context_length': max_context or 32768,
                    'qwen2.attention.head_count': 32,
                    'qwen2.attention.head_count_kv': 32,
                    'qwen2.embedding_length': 4096, # 27B Hybrid Default
                    'qwen2.feed_forward_length': 17408, # 27B Hybrid Default
                    'qwen2.block_count': 64, # 27B Hybrid Default
                    'qwen2.attention.layer_norm_rms_epsilon': 1e-6,
                    'tokenizer.ggml.tokens': [""] * 248320, 
                    'qwen2.rope.freq_base': 1000000.0,
                }
        else:
            trace("Loading as GGUF")
            kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)

        trace(f"Load finished. Keys in state_dict: {len(state_dict)}")
        
        if path_str.endswith(".safetensors"):
            trace("Normalizing Safetensors keys to GGUF format")
            new_sd = {}
            for k, v in state_dict.items():
                nk = k.replace('model.embed_tokens.weight', 'token_embd.weight') \
                      .replace('model.norm.weight', 'output_norm.weight') \
                      .replace('lm_head.weight', 'output.weight') \
                      .replace('model.layers.', 'blk.') \
                      .replace('language_blk.', '') \
                      .replace('.input_layernorm.', '.attn_norm.') \
                      .replace('.post_attention_layernorm.', '.ffn_norm.') \
                      .replace('.self_attn.q_proj.', '.attn_q.') \
                      .replace('.self_attn.k_proj.', '.attn_k.') \
                      .replace('.self_attn.v_proj.', '.attn_v.') \
                      .replace('.self_attn.o_proj.', '.attn_output.') \
                      .replace('.mlp.gate_proj.', '.ffn_gate.') \
                      .replace('.mlp.up_proj.', '.ffn_up.') \
                      .replace('.mlp.down_proj.', '.ffn_down.')
                new_sd[nk] = v
            state_dict = new_sd

        state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}
        
        if 'output.weight' not in state_dict:
            emb_key = 'token_embd.weight' if 'token_embd.weight' in state_dict else 'model.embed_tokens.weight'
            trace(f"Generating output.weight from {emb_key}")
            state_dict['output.weight'] = state_dict[emb_key]

        arch = kv.get('general.architecture', 'qwen2') 
        trace(f"Using architecture: {arch}")

        max_context = min(max_context, kv.get(f'{arch}.context_length', 32768)) if max_context is not None else kv.get(f'{arch}.context_length', 32768)
        n_heads = kv.get(f'{arch}.attention.head_count', 32)
        n_kv_heads = kv.get(f'{arch}.attention.head_count_kv', 32)
        
        ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')}) if arch in ('qwen35', 'qwen35moe') and f'{arch}.ssm.conv_kernel' in kv else None
        
        if arch in ('qwen35', 'qwen35moe', 'glm4moe'): 
            state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}
        
        embedding_length = kv.get(f'{arch}.embedding_length', 4096)
        hidden_dim = kv.get(f'{arch}.feed_forward_length', 17408)
        num_blocks = kv.get(f'{arch}.block_count', 64)
        head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', embedding_length // n_heads))

        trace(f"Configuring model: dim={embedding_length}, hidden_dim={hidden_dim}, blocks={num_blocks}")

        config = TransformerConfig(
            num_blocks=num_blocks, 
            dim=embedding_length, 
            hidden_dim=hidden_dim, 
            n_heads=n_heads, 
            n_kv_heads=n_kv_heads, 
            norm_eps=kv.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-6), 
            vocab_size=len(kv.get('tokenizer.ggml.tokens', [""] * 248320)), 
            head_dim=head_dim, 
            rope_theta=kv.get(f'{arch}.rope.freq_base', 1000000.0), 
            rope_dim=kv.get(f'{arch}.rope.dimension_count', head_dim), 
            v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)), 
            max_context=max_context, 
            qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0, 
            num_experts=kv.get(f'{arch}.expert_count', 0), 
            num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0), 
            norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe')), 
            kv_lora_rank=kv.get(f'{arch}.attention.kv_lora_rank', 0), 
            q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0), 
            leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0), 
            shared_expert_dim=kv.get(f'{arch}.expert_shared_feed_forward_length', 0), 
            shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict, 
            dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0, 
            routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), 
            attn_output_gate=arch in ('qwen35', 'qwen35moe'), 
            ssm=ssm, 
            full_attention_interval=kv.get(f'{arch}.full_attention_interval', 0), 
            qkv_bias='blk.0.attn_q.bias' in state_dict, 
            expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict
        )

        model = Transformer(config)
        trace("Mapping weights to devices manually...")
        new_state_dict = {}
        for k, v in state_dict.items():
            target = "AMD:0"
            layer_match = re.search(r'blk\.(\d+)', k)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                # Offload only the top 8 layers to METAL as requested
                if layer_idx >= (num_blocks - 8): target = "METAL"
            new_state_dict[k] = v.to(target)
            
        trace("Calling nn.state.load_state_dict")
        nn.state.load_state_dict(model, new_state_dict, verbose=False, consume=True, realize=False)
        if realize:
            trace("Realizing parameters")
            for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
            Tensor.realize(*params)
        trace("Transformer.from_gguf exiting")
        return model, kv

    def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
        trace(f"Generation started (temp={temperature})")
        v_start_pos, v_toks, temp = UOp.variable("start_pos", 0, self.max_context-1), UOp.variable("toks", 1, chunk_size), Tensor(temperature).contiguous()
        t, start_pos = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context), 0
        out, prompt_len = None, len(tokens)
        while len(tokens) < self.max_context:
            sp, nt = v_start_pos.bind(start_pos), v_toks.bind(min(chunk_size, len(tokens) - start_pos))
            out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
            start_pos += nt.val
            if start_pos < len(tokens): continue
            tokens.append(int(out.item()))
            yield tokens[-1]

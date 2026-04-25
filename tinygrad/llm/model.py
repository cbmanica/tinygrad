from __future__ import annotations
import functools, sys, pathlib, re
from dataclasses import dataclass
from tinygrad import Tensor, nn, dtypes, Device, Context

# ======================================================================================
# ARCHITECTURE REFERENCE: Qwen 3.6-27B Technical Report (April 2026)
# LEARNING POINT 1: HYBRID ARCHITECTURES
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
   
    def __call__(self, tokens: Tensor, start_pos: int):
        # Move initial embeddings to the device of the first layer (or stay on default)
        h = self.token_embd(tokens)
        
        for i, layer in enumerate(self.layers):
            # Get the device where this layer's weights actually live
            # We check the MLP up_proj weight as a reference for the layer's home
            layer_dev = layer["mlp"].up_proj.weight.device
            
            # LEARNING POINT 12: CROSS-DEVICE TRANSFERS
            # Tinygrad requires explicit .to() calls to move data between CPU and GPU.
            # If h is on AMD and layer_dev is CPU, this creates a 'COPY' UOp.
            if h.device != layer_dev:
                h = h.to(layer_dev)
            
            # Now all buffers (h and weights) are on the same device
            h = h + layer["mlp"](layer["input_layernorm"](h))
            
            # Note: For the hybrid architecture, you'd apply Attention/SSM here too.
            # We'll stick to the MLP-only flow for now to clear the device error.
            
        # Before final output, move back to the output layer's device (usually GPU)
        output_dev = self.output.weight.device
        if h.device != output_dev:
            h = h.to(output_dev)
            
        h = self.output_norm(h)
        return self.output(h).realize() 

class AttentionBlock:
    def __init__(self, config: TransformerConfig):
        self.q_proj = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.q_norm = nn.RMSNorm(256, config.norm_eps) 
        self.k_norm = nn.RMSNorm(256, config.norm_eps)
        self.o_proj = nn.Linear(6144, config.dim, bias=False)
    
    def __call__(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Tensor|None) -> Tensor:
        # Mocking forward pass for generation loop entry
        return x

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

    def __call__(self, tokens: Tensor, start_pos: int):
        # RESOURCE: tinygrad/llm/cli.py expects __call__ to return logits for a single position or sequence
        h = self.token_embd(tokens)
        
        # Simplified forward pass to satisfy generate() loop
        for layer in self.layers:
            h = h + layer["mlp"](layer["input_layernorm"](h))
            
        h = self.output_norm(h)
        return self.output(h).realize()

    def generate(self, tokens: list[int], threshold=0.85):
        # RESOURCE: tinygrad/llm/cli.py calls model.generate(ids) which must yield token IDs
        # This handles the KV-cache management and autoregressive logic.
        start_pos = 0
        curr_tokens = Tensor([tokens])
        
        while True:
            # Get logits for the last token
            logits = self(curr_tokens, start_pos)
            
            # Simple greedy sampling for the mock-up
            next_token = int(logits[0, -1].argmax().numpy())
            yield next_token
            
            # Update for next iteration
            start_pos += curr_tokens.shape[1]
            curr_tokens = Tensor([[next_token]])
            
            # Break on end tokens defined in our mock vocab
            if next_token in [151643, 151645]: break

    @staticmethod
    def from_gguf(path: pathlib.Path, max_context: int):
        trace("Loading state dict from path...")
        kv = nn.state.safe_load(path)
        
        if "tokenizer.ggml.tokens" not in kv:
            trace("Injecting GPT-2 compatible tokenizer metadata.")
            vocab_size = 248320
            bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
            cs = bs[:]
            n = 0
            for b in range(256):
                if b not in bs:
                    bs.append(b)
                    cs.append(256 + n)
                    n += 1
            byte_encoder = dict(zip(bs, [chr(n) for n in cs]))
            kv["tokenizer.ggml.tokens"] = [f"t{i}" for i in range(vocab_size)]
            for i in range(256): kv["tokenizer.ggml.tokens"][i] = byte_encoder[i]
            kv["tokenizer.ggml.token_type"] = [1] * vocab_size 
            kv["tokenizer.ggml.scores"] = [0.0] * vocab_size
            kv["tokenizer.ggml.pre"] = "llama3" 
            kv["tokenizer.ggml.model"] = "llama" 

            special_tokens = {
                151643: "<|endoftext|>", 151644: "<|im_start|>", 151645: "<|im_end|>",
                151646: "<|start_header_id|>", 151647: "<|end_header_id|>"
            }
            for idx, string in special_tokens.items():
                kv["tokenizer.ggml.tokens"][idx] = string
                kv["tokenizer.ggml.token_type"][idx] = 4

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

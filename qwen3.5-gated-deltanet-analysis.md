# Qwen3.5 Gated DeltaNet Linear Attention — Technical Analysis

**Purpose**: Research for proposing new ONNX operators to support Qwen3.5/Qwen3-Next linear attention.

**Date**: 2026-02-27

**Sources**:
- HuggingFace transformers: `src/transformers/models/qwen3_next/modular_qwen3_next.py`
- HuggingFace transformers: `src/transformers/models/qwen3_5/modular_qwen3_5.py`
- flash-linear-attention library: `fla/ops/gated_delta_rule/`
- DeltaNet paper: [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) (Yang et al., 2024)
- Gated DeltaNet paper: [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) (Yang et al., ICLR 2025)

---

## 1. Architecture Overview

Qwen3.5 (HuggingFace `model_type: "qwen3_5"`) inherits from **Qwen3-Next** (`model_type: "qwen3_next"`). It is a **hybrid architecture** that interleaves two types of token-mixing layers:

1. **Full attention layers** — Standard softmax attention with GQA and RoPE
2. **Linear attention layers** — **Gated DeltaNet** (a gated variant of the delta rule)

The layer pattern is configured by `layer_types` in the config. By default, every 4th layer is full attention (controlled by `full_attention_interval`):
```
[linear, linear, linear, full_attention, linear, linear, linear, full_attention, ...]
```

This means ~75% of layers are linear attention and ~25% are full softmax attention. The full attention layers provide global context and strong retrieval capability, while linear layers provide efficient O(1)-per-token inference.

Qwen3.5 is also a **Vision-Language Model** (inheriting VL capabilities from Qwen3-VL) with **Mixture-of-Experts** MLP layers (from Qwen2-MoE).

---

## 2. Mathematical Formulation

### 2.1 Background: Standard Attention

```
O = softmax(QK^T / √d_k) V
```
- Complexity: O(n² · d) where n = sequence length
- No fixed-size state — must attend to all previous tokens

### 2.2 Background: Linear Attention (Katharopoulos et al., 2020)

Replace softmax with a feature map φ:
```
O_t = Σ_{i≤t} φ(q_t)^T φ(k_i) v_i / Σ_{i≤t} φ(q_t)^T φ(k_i)
```

Recurrent form using state matrix S:
```
S_t = S_{t-1} + k_t ⊗ v_t       (additive outer product update)
o_t = q_t^T S_t                   (read with query)
```
- State shape: (d_k × d_v) per head — fixed regardless of sequence length
- Complexity: O(n · d²) — linear in sequence length
- **Problem**: Additive updates mean the state grows without bound and can't forget

### 2.3 Background: DeltaNet (Yang et al., 2024)

Replace additive update with the **delta rule** from associative memory theory:
```
S_t = S_{t-1} + β_t · k_t ⊗ (v_t - S_{t-1}^T k_t)
o_t = q_t^T S_t / √d_k
```

The term `(v_t - S_{t-1}^T k_t)` is the **prediction error**: the difference between the new value and what the state currently predicts for this key. Instead of blindly adding to memory, DeltaNet corrects the existing memory — much better at in-context retrieval.

### 2.4 Gated DeltaNet (Yang et al., ICLR 2025) — What Qwen3.5 Uses

Combines gating (exponential decay) with the delta rule:

#### Recurrent Form (used during autoregressive inference)

```
g_t = exp(α_t)                              // per-head scalar decay gate
β_t = σ(b_t)                                // per-head scalar update rate (sigmoid)
q̃_t = l2norm(q_t),  k̃_t = l2norm(k_t)     // L2-normalized Q and K

S_t = g_t · S_{t-1} + k̃_t ⊗ [β_t · (v_t − (g_t · S_{t-1})^T k̃_t)]
o_t = q̃_t^T · S_t / √d_k
```

Where the gating parameter is computed as:
```
α_t = -exp(A_log) · softplus(a_t + dt_bias)
g_t = exp(α_t)  ∈ (0, 1]
```

- `A_log` ∈ R^{num_heads}: Learned log-space decay rate (initialized uniform in [0, 16])
- `dt_bias` ∈ R^{num_heads}: Learned time-step bias (initialized to 1)
- `a_t` ∈ R^{num_heads}: Input-dependent component (from `in_proj_ba`)
- `b_t` ∈ R^{num_heads}: Input-dependent beta (from `in_proj_ba`)

The decay g_t ∈ (0, 1] controls how much old memory is retained. When g_t ≈ 0, old memory is erased. When g_t ≈ 1, old memory is fully retained. This is **data-dependent** — the model learns when to forget.

#### Expanded Recurrent Step (from code)

```python
# 1. Decay old state
S = S * g_t                        # (num_heads, d_k, d_v) * (num_heads, 1, 1)

# 2. Retrieve what state predicts for this key
kv_mem = (S * k_t[..., None]).sum(dim=-2)  # (num_heads, d_v)

# 3. Compute delta (error correction)
delta = (v_t - kv_mem) * β_t       # (num_heads, d_v) * (num_heads, 1)

# 4. Write correction into state
S = S + k_t[..., None] * delta[..., None]  # outer product update

# 5. Read output
o_t = (S * q_t[..., None]).sum(dim=-2)     # (num_heads, d_v)
```

#### Chunk-Parallel Form (used during prefill / training)

For processing sequences in parallel (e.g., prefill), the sequence is divided into chunks of size C (default: 64). Within each chunk, the algorithm uses a WY-like decomposition to compute the effect of multiple delta-rule updates in parallel.

Key steps per chunk i:
```
1. Compute cumulative decay within chunk:
   G[j] = Σ_{m=0}^{j} g[m]  (cumsum of log-decay)

2. Compute decay-weighted attention matrix within chunk:
   L[j,k] = exp(G[j] - G[k]) for j ≥ k, 0 otherwise (lower triangular)

3. WY decomposition for delta correction:
   A = -(k_β @ k^T * L), masked upper triangle
   Forward-substitute to get correction matrix
   v_corrected = (I + A) @ (v * β)

4. Cross-chunk state update:
   S_i = decay * S_{i-1} + k^T @ v_corrected
   o_inter = q * exp(G) @ S_{i-1}    (inter-chunk attention)
   o_intra = (q @ k^T * L) @ v_new   (intra-chunk attention)
   o = o_inter + o_intra
```

Complexity: O(n · C · d²) — linear in sequence length n.

---

## 3. Full Forward Pass of Qwen3NextGatedDeltaNet

### 3.1 Projections

```
x: (batch, seq_len, hidden_size)

# Project to Q, K, V, Z (output gate)
qkvz = in_proj_qkvz(x)    # Linear: hidden_size → key_dim*2 + value_dim*2

# Project to β and α (for gates)
ba = in_proj_ba(x)         # Linear: hidden_size → num_v_heads*2
```

### 3.2 Interleaved Head Layout (GQA-style for linear attention)

The projections use a **grouped layout** where K heads are fewer than V heads:
```
num_k_heads = 16 (default)
num_v_heads = 32 (default)
key_head_dim = 128
value_head_dim = 128
```

Q and K share the same head count and dimension (num_k_heads × key_head_dim = 2048).
V has more heads (num_v_heads × value_head_dim = 4096).
K heads are repeated to match V heads via repeat_interleave(num_v_heads // num_k_heads).

### 3.3 Causal Conv1D

After splitting Q, K, V from the projection, they are concatenated and passed through a **depthwise causal 1D convolution**:
```
mixed_qkv: (batch, key_dim*2 + value_dim, seq_len)  // after transpose
conv1d: depthwise, kernel=4, groups=conv_dim, padding=kernel-1
→ SiLU activation
→ Split back to Q, K, V
```

Purpose: Short-range local context mixing before the linear attention. This replaces positional encodings (RoPE is NOT used for linear attention layers). The conv provides a local receptive field of 4 tokens.

### 3.4 Gate Computation

```
β = sigmoid(b)                              # (batch, seq, num_v_heads)
g = -A_log.exp() * softplus(a + dt_bias)    # (batch, seq, num_v_heads)
```

### 3.5 Core Gated Delta Rule

```
# L2 normalize Q and K
q = l2norm(q)    # (batch, seq, num_v_heads, key_head_dim)
k = l2norm(k)    # (batch, seq, num_v_heads, key_head_dim)

# Prefill: chunk algorithm
output, final_state = chunk_gated_delta_rule(q, k, v, g, β)
# OR
# Decode: recurrent algorithm
output, final_state = recurrent_gated_delta_rule(q, k, v, g, β, initial_state)
```

### 3.6 Gated RMSNorm + Output Projection

```
# Gated RMSNorm: normalize then gate with z
output = rms_norm(output) * silu(z)     # z was from the initial qkvz projection

# Project back to hidden_size
output = out_proj(output)               # Linear: value_dim → hidden_size
```

---

## 4. Primitive Operations and Their Tensor Shapes

Assuming Qwen3.5-9B defaults: hidden=4096, key_dim=2048, value_dim=4096, num_k_heads=16, num_v_heads=32, key_head_dim=128, value_head_dim=128.

| # | Operation | Input Shape(s) | Output Shape | ONNX Equivalent |
|---|-----------|---------------|--------------|-----------------|
| 1 | `in_proj_qkvz` Linear | (B,T,4096) | (B,T,12288) | MatMul + (optional bias) ✅ |
| 2 | `in_proj_ba` Linear | (B,T,4096) | (B,T,64) | MatMul ✅ |
| 3 | Reshape + Split qkvz | (B,T,12288) | Q(B,T,16,128), K(B,T,16,128), V(B,T,32,128), Z(B,T,32,128) | Reshape + Split ✅ |
| 4 | Split ba | (B,T,64) | β_raw(B,T,32), α_raw(B,T,32) | Split ✅ |
| 5 | Concatenate Q,K,V | Q,K,V flattened | (B, 8192, T) after transpose | Concat + Transpose ✅ |
| 6 | **CausalConv1D** | (B, 8192, T) | (B, 8192, T) | **⚠️ No fused op** — decomposable to Conv+Pad+SiLU |
| 7 | Split post-conv | (B, 8192, T) | Q(B,T,2048), K(B,T,2048), V(B,T,4096) | Split ✅ |
| 8 | **L2Norm** on Q,K | (B,T,H,128) | (B,T,H,128) | **⚠️ Decomposable**: x * rsqrt(reducesumsquare + eps) |
| 9 | Sigmoid(β_raw) | (B,T,32) | (B,T,32) | Sigmoid ✅ |
| 10 | Exp(-A_log) | (32,) | (32,) | Neg + Exp ✅ |
| 11 | Softplus(α_raw + dt_bias) | (B,T,32) | (B,T,32) | Softplus ✅ |
| 12 | g = -exp(A_log) * softplus(...) | (B,T,32) | (B,T,32) | Mul + Neg ✅ |
| 13 | repeat_interleave K heads | (B,T,16,128) | (B,T,32,128) | Expand/Tile ✅ |
| 14 | **GatedDeltaRuleChunk** (prefill) | Q,K,V(B,T,32,128), g,β(B,T,32) | (B,T,32,128) + state(B,32,128,128) | **❌ NO EQUIVALENT** |
| 15 | **GatedDeltaRuleRecurrent** (decode) | Q,K,V(B,1,32,128), g,β(B,1,32), state(B,32,128,128) | (B,1,32,128) + state(B,32,128,128) | **❌ NO EQUIVALENT** |
| 16 | **GatedRMSNorm** | (B·T, 128), gate(B·T, 128) | (B·T, 128) | **⚠️ Decomposable**: RMSNorm(x) * SiLU(gate) |
| 17 | `out_proj` Linear | (B,T,4096) | (B,T,4096) | MatMul ✅ |

---

## 5. Cache / State During Autoregressive Generation

### 5.1 Hybrid Cache Structure (`Qwen3NextDynamicCache`)

The model uses a **heterogeneous cache** where different layer types have different state:

**Full attention layers** (standard KV cache):
```
key_cache[layer]:   (batch, num_kv_heads, seq_len, head_dim)  — grows with sequence
value_cache[layer]: (batch, num_kv_heads, seq_len, head_dim)  — grows with sequence
conv_states[layer]:      None
recurrent_states[layer]: None
```

**Linear attention layers** (fixed-size state):
```
key_cache[layer]:        None
value_cache[layer]:      None
conv_states[layer]:      (batch, conv_dim, conv_kernel_size)  — fixed size (B, 8192, 4)
recurrent_states[layer]: (batch, num_v_heads, key_head_dim, value_head_dim) — fixed size (B, 32, 128, 128)
```

### 5.2 Inference Flow

**Prefill (first forward pass with full prompt)**:
1. Full attention layers: compute KV cache as usual
2. Linear attention layers: run `chunk_gated_delta_rule` → store `final_state` as `recurrent_states[layer]`
3. Conv state: padded input stored as `conv_states[layer]`

**Decode (subsequent tokens, one at a time)**:
1. Full attention layers: append to KV cache, run standard attention
2. Linear attention layers: run `recurrent_gated_delta_rule` with `initial_state = recurrent_states[layer]` → update state
3. Conv state: `causal_conv1d_update` incrementally updates the sliding window

### 5.3 Memory Advantage

For a Qwen3.5-9B with 32 layers (24 linear + 8 full attention):
- **Linear layers**: 24 × (B × 32 × 128 × 128) = 24 × 512KB per batch = **12MB fixed** (fp16)
- **Full attention layers**: 8 × standard KV cache — grows with sequence length

This is dramatically less memory than 32 full attention layers for long sequences.

---

## 6. Key Differences from Standard Scaled Dot-Product Attention

| Aspect | Standard SDPA | Gated DeltaNet |
|--------|--------------|----------------|
| **Complexity** | O(n²d) per layer | O(nd²) recurrent / O(nCd²) chunk |
| **State** | KV cache grows with seq_len | Fixed-size matrix S ∈ R^{d_k×d_v} |
| **Causal masking** | Explicit mask matrix | Implicit (recurrent structure) |
| **Position encoding** | RoPE applied to Q,K | CausalConv1D (local, no absolute positions) |
| **Memory mechanism** | Full pairwise attention weights | Delta rule (error-correcting updates) |
| **Gating** | None (softmax normalization) | Exponential decay g + update rate β |
| **Normalization** | Softmax over sequence | L2 norm on Q and K |
| **Output gating** | None | Sigmoid gate on full-attention; SiLU-gated RMSNorm on linear |
| **KV head grouping** | GQA (fewer KV heads) | Separate key/value head counts (num_k_heads ≠ num_v_heads) |

---

## 7. Novel Operations Requiring New ONNX Operators

### 7.1 **GatedDeltaRuleRecurrent** (CRITICAL — needed for decode)

The single-step recurrent update:
```
Inputs:
  q:     (B, 1, H, d_k)     — query
  k:     (B, 1, H, d_k)     — key (L2-normalized)
  v:     (B, 1, H, d_v)     — value
  g:     (B, 1, H)           — decay gate (log-space)
  beta:  (B, 1, H)           — update rate
  state: (B, H, d_k, d_v)   — recurrent state matrix

Outputs:
  output:    (B, 1, H, d_v)
  new_state: (B, H, d_k, d_v)

Algorithm:
  state = exp(g) * state
  retrieved = einsum('bhkv,bhk->bhv', state, k)
  delta = beta * (v - retrieved)
  state = state + einsum('bhk,bhv->bhkv', k, delta)
  output = einsum('bhkv,bhk->bhv', state, q) / sqrt(d_k)
```

**Why a dedicated op?** This involves a sequence of tightly coupled matrix operations on a state tensor. Decomposing into individual ONNX ops would require materializing intermediate tensors of shape (B, H, d_k, d_v) multiple times, losing the opportunity for kernel fusion. A fused operator can:
1. Keep the state in registers/shared memory
2. Avoid multiple global memory round-trips
3. Fuse the decay, retrieval, update, and read in one kernel

### 7.2 **GatedDeltaRuleChunk** (IMPORTANT — needed for prefill)

The chunk-parallel algorithm for processing multiple tokens at once:
```
Inputs:
  q:     (B, T, H, d_k)
  k:     (B, T, H, d_k)
  v:     (B, T, H, d_v)
  g:     (B, T, H)
  beta:  (B, T, H)
  initial_state: (B, H, d_k, d_v) or None
  chunk_size: int (default 64)

Outputs:
  output: (B, T, H, d_v)
  final_state: (B, H, d_k, d_v)
```

This is algorithmically complex (WY decomposition, decay masks, inter/intra-chunk computation) but critical for efficient prefill. Without this, prefill would fall back to the recurrent form at O(n·d²) with sequential processing.

### 7.3 **CausalConv1DWithSiLU** (NICE TO HAVE — shared with Mamba)

```
Inputs:
  x:          (B, D, T)      — input tensor
  weight:     (D, 1, K)      — depthwise conv weights
  bias:       (D,) or None
  conv_state: (B, D, K) or None  — for incremental decode

Outputs:
  output:     (B, D, T)
  new_state:  (B, D, K) or None
```

This is already needed for Mamba/SSM models. Can be decomposed into Conv1D + padding + SiLU, but the fused version (from `causal-conv1d` library) is significantly faster.

### 7.4 **L2Normalize** (NICE TO HAVE)

```
output = x * rsqrt(sum(x², dim=-1, keepdim=True) + eps)
```

Decomposable into existing ops but a dedicated op would be cleaner and more efficient.

### 7.5 **GatedRMSNorm** (NICE TO HAVE)

```
output = RMSNorm(x, weight) * SiLU(gate)
```

Decomposable but the fused version from FLA is notably faster.

---

## 8. Comparison with Known Linear Attention Variants

| Model | State Update Rule | State Shape | Gating | Key Innovation | Used In |
|-------|-------------------|-------------|--------|---------------|---------|
| Linear Attention | S += k⊗v | H×d_k×d_v | None | Remove softmax | — |
| RetNet | S = γ·S + k⊗v | H×d_k×d_v | Fixed exponential decay | Multi-scale retention | — |
| GLA | S = diag(G)·S + k⊗v | H×d_k×d_v | Data-dependent matrix | Gated linear attention | — |
| DeltaNet | S += k⊗β(v − S^Tk) | H×d_k×d_v | None | Delta rule error correction | — |
| **Gated DeltaNet** | **S = g·S + k⊗β(v − gS^Tk)** | **H×d_k×d_v** | **Data-dependent scalar** | **Delta rule + gating** | **Qwen3.5, Qwen3-Next** |
| Mamba (S6) | h = Āh + B̄x | H×d_state | Selective params | Selection mechanism | Jamba, FalconMamba |
| Mamba2 (SSD) | h = Āh + B̄x | H×d_state×d_head | Structured state space dual | State space duality | — |
| RWKV-6 | h = diag(w)·h + k⊗v | H×d_k×d_v | Channel-wise decay | Token/channel mixing | RWKV |
| HGRN2 | h = f·h + i·x | H×d | Forget/input gates | Hierarchical gating | — |

### Key Insight: Why Gated DeltaNet is Architecturally Superior

1. **Delta rule > additive update**: The error-correction mechanism means the state can accurately store and retrieve individual key-value pairs, critical for in-context learning and retrieval tasks.

2. **Gating > no gating**: The exponential decay allows the model to forget irrelevant context, preventing state saturation over long sequences.

3. **Combined > either alone**: Gating handles bulk erasure; delta rule handles precise writes. Together they provide both coarse and fine-grained memory control.

4. **Hybrid with full attention**: The few interleaved full-attention layers (25%) provide global O(n²) retrieval for the hardest cases, while linear layers handle the routine O(1) processing.

---

## 9. ONNX Operator Landscape and Gap Analysis

### 9.1 Current State of ONNX Attention Support

Based on comprehensive research of the ONNX standard spec and ORT contrib ops:

| Existing Op | Domain | What It Does | Linear Attention? |
|-------------|--------|--------------|-------------------|
| `Attention` (opset 23) | Standard | softmax(QK^T/√d)V | ❌ Softmax only |
| `MultiHeadAttention` | com.microsoft | Multi-head softmax attention | ❌ Softmax only |
| `GroupQueryAttention` | com.microsoft | GQA softmax attention | ❌ Softmax only |
| `PagedAttention` | com.microsoft | Paged KV cache + softmax | ❌ Softmax only |
| `SparseAttention` | com.microsoft | Sparse patterns + softmax | ❌ Softmax only |
| `DecoderMaskedMultiHeadAttention` | com.microsoft | Optimized decode softmax | ❌ Softmax only |
| `FlexAttention` (proposed, onnx/onnx#7494) | Standard | Custom subgraphs for attention | ❌ Still softmax-based |
| `Scan` (opset 25) | Standard | General recurrence | ⚠️ Technically possible but no fusion |
| `CumSum` (opset 14) | Standard | Scalar cumulative sum | ⚠️ Only scalar, not matrix cumulative products |

**Key finding: ZERO operators exist for linear attention, SSMs, or any non-softmax token mixing.** All 12+ attention-related ops in ONNX + ORT are exclusively softmax-based. This is a completely greenfield area.

### 9.2 Why ONNX Scan Is Insufficient

The ONNX `Scan` op can technically represent the GatedDeltaNet recurrence by iterating over the sequence. However:

1. **No kernel fusion**: Each iteration executes ~15-20 separate ONNX ops (Mul, ReduceSum, Sub, Add, etc.), each requiring a GPU kernel launch
2. **Memory bandwidth**: Intermediate tensors of shape (B, H, d_k, d_v) are materialized to global memory and read back ~5 times per step
3. **No hardware optimization**: A fused Triton/CUDA kernel (like `fla.ops.gated_delta_rule`) can keep the state in registers/shared memory
4. **Estimated performance gap**: 10-50x slower than a fused kernel, based on the ratio of memory bandwidth to compute for this operation class

### 9.3 Proposed ONNX Operators

#### Priority 1: `LinearAttentionRecurrent` (CRITICAL — needed for decode)

A **generalized** recurrent linear attention step that can support multiple update rules:

```
// Generalized linear attention recurrent step
LinearAttentionRecurrent(
    query,           // (B, 1, H, d_k)
    key,             // (B, 1, H, d_k)
    value,           // (B, 1, H, d_v)
    state,           // (B, H, d_k, d_v) — recurrent state
    gate=None,       // (B, 1, H) — exponential decay (optional)
    beta=None,       // (B, 1, H) — update rate (optional)
    mode="gated_delta"  // string attribute
) -> (output, new_state)
    // output:    (B, 1, H, d_v)
    // new_state: (B, H, d_k, d_v)
```

Supported modes (string enum — extensible for future variants):
- `"linear"`: S += k⊗v; o = q^T S (vanilla linear attention)
- `"gated"`: S = g·S + k⊗v; o = q^T S (GLA / RetNet style)
- `"delta"`: S += k⊗β(v − S^Tk); o = q^T S (DeltaNet)
- `"gated_delta"`: S = g·S + k⊗β(v − g·S^Tk); o = q^T S (Gated DeltaNet — Qwen3.5)

**Why generalize?** All these variants share the same **state interface contract**:
- **State shape**: (B, H, d_k, d_v) — fixed-size matrix per head, independent of sequence length
- **Inputs per step**: query, key, value (all per-token), plus optional gate and beta scalars
- **Outputs per step**: output vector + updated state

The mode attribute selects the internal update rule, but backends only need to implement the I/O contract. This is analogous to how `LSTM`, `GRU`, and `RNN` are separate modes of recurrent processing with the same external interface pattern. Using a string enum (not int) ensures future variants (RetNet-v2, RWKV-6, etc.) can be added without breaking the schema.

**Models this enables**: Qwen3.5/Qwen3-Next, Gated DeltaNet, DeltaNet, GLA, RetNet, and future linear attention variants.

#### Priority 2: `LinearAttentionChunk` (CRITICAL — needed for prefill)

Chunk-parallel computation for processing multiple tokens efficiently:

```
LinearAttentionChunk(
    query,           // (B, T, H, d_k)
    key,             // (B, T, H, d_k)
    value,           // (B, T, H, d_v)
    gate=None,       // (B, T, H) — log-space decay
    beta=None,       // (B, T, H) — update rate
    initial_state=None,  // (B, H, d_k, d_v)
    chunk_size=64,   // int attribute
    mode="gated_delta"
) -> (output, final_state)
    // output:      (B, T, H, d_v)
    // final_state: (B, H, d_k, d_v)
```

Without this, prefill falls back to sequential recurrent processing — O(T) sequential steps instead of O(T/C) chunks processed with internal parallelism.

#### Priority 3: `CausalConv1D` (IMPORTANT — shared with Mamba)

```
CausalConv1D(
    input,       // (B, D, T)
    weight,      // (D, 1, K) — depthwise
    bias=None,   // (D,)
    conv_state=None,  // (B, D, K-1) — for incremental decode
    activation="silu"  // fused activation
) -> (output, new_state)
```

Already needed for Mamba/Jamba/FalconMamba. A shared op benefits the entire SSM + linear attention ecosystem.

#### Priority 4: `GatedRMSNorm`, `L2Normalize` (NICE TO HAVE)

Nice-to-have fusions. Can be decomposed into existing ops at moderate cost (~2-3 ops each).

---

## 10. Detailed Operation Decomposition for Fallback

If we can't get new ONNX ops, here's how to decompose the gated delta rule recurrent step into existing ONNX ops:

```
# All shapes include batch (B) and heads (H) dimensions

# Step 1: Decay state
# g: (B, 1, H) → exp → (B, 1, H) → unsqueeze → (B, H, 1, 1)
gate = Exp(g)                          # existing op
gate = Unsqueeze(gate, [-1, -2])       # existing op
state = Mul(state, gate)               # (B, H, dk, dv) * (B, H, 1, 1)

# Step 2: Retrieve
# k: (B, 1, H, dk) → transpose → (B, H, dk, 1)
k_t = Transpose(k)
k_expanded = Unsqueeze(k_t, [-1])     # (B, H, dk, 1)
# state * k_expanded → (B, H, dk, dv) * (B, H, dk, 1) → sum over dk
retrieved = ReduceSum(Mul(state, k_expanded), axis=-2)  # (B, H, dv)

# Step 3: Delta
delta = Sub(v, retrieved)              # (B, H, dv)
beta_expanded = Unsqueeze(beta, [-1])  # (B, H, 1)
delta = Mul(delta, beta_expanded)      # (B, H, dv)

# Step 4: Write
# k: (B, H, dk, 1) * delta: (B, H, 1, dv) → outer product
delta_expanded = Unsqueeze(delta, [-2])  # (B, H, 1, dv)
update = Mul(k_expanded, delta_expanded) # (B, H, dk, dv)
state = Add(state, update)              # (B, H, dk, dv)

# Step 5: Read
q_expanded = Unsqueeze(q_t, [-1])      # (B, H, dk, 1)
output = ReduceSum(Mul(state, q_expanded), axis=-2)  # (B, H, dv)
scale = Sqrt(Constant(dk))
output = Div(output, scale)
```

This requires ~15 ONNX ops per recurrent step vs. 1 fused op. The fused version would be 5-10x faster due to reduced memory bandwidth.

---

## 11. Summary

Qwen3.5/Qwen3-Next uses **Gated DeltaNet** — a state-of-the-art linear attention mechanism that combines:
- **Delta rule** for error-correcting memory updates (superior retrieval)
- **Exponential gating** for adaptive memory decay (prevents saturation)
- **Causal Conv1D** for local context (replaces positional encoding)
- **L2 normalization** on Q/K (replaces softmax normalization)
- **Hybrid architecture** with interleaved full-attention layers

The core operation that needs new ONNX support is the **gated delta rule state update**, which operates on a fixed-size state matrix S ∈ R^{d_k × d_v} per head. This is fundamentally different from softmax attention and cannot be efficiently expressed with existing ONNX operators.

### Proposed New ONNX Operators (Priority Order)

1. **`LinearAttentionRecurrent`** — Generalized recurrent step (supports linear/gated/delta/gated_delta modes)
2. **`LinearAttentionChunk`** — Chunk-parallel computation for prefill
3. **`CausalConv1D`** — Fused depthwise causal convolution (shared with Mamba/SSM models)
4. **`GatedRMSNorm`** — RMSNorm with SiLU gating (minor)
5. **`L2Normalize`** — L2 normalization along a dimension (minor)

The first two are **essential** for competitive performance. Without them, inference requires decomposition into ~15-20 primitive ONNX ops per recurrent step (10-50x slower than a fused kernel). The ONNX Scan op is technically usable but provides no kernel fusion, making it similarly slow.

**Landscape context**: As of 2026-02, ZERO ONNX operators exist for any non-softmax attention variant. All 12+ existing attention ops (standard + ORT contrib) are exclusively softmax-based. The FlexAttention proposal (onnx/onnx#7494) also only extends softmax attention. These proposals would be the first operators enabling efficient linear attention in ONNX, benefiting the growing family of hybrid models: Qwen3.5, Qwen3-Next, OLMo-Hybrid, and future architectures using GLA, DeltaNet, RetNet, or similar linear attention variants.

# AMD Multi-Device Inference Notes (Qwen3.6-27B on RX 7900 XTX + Metal)

## Status: Working

AMD acceleration is enabled by default (`amd_budget_gb=20.0`). The smoke test passes with 52 AMD layers + 12 Metal layers.

## Architecture

For a machine with an AMD RX 7900 XTX (24 GB, connected via USB4/Thunderbolt) alongside an Apple Silicon Mac (Metal, unified memory), `from_safetensors` in `tinygrad/llm/safetensors_loader.py` distributes transformer blocks across devices:

- Full-attention blocks (every `full_attention_interval`-th block, default 4) → AMD first (highest memory priority)
- Remaining blocks → AMD until the `amd_budget_gb` budget is filled, then Metal
- Non-block tensors (`token_embd`, `output`, `output_norm`) → Metal

## Key Implementation Details

### 1. COMGR prewarm (most subtle)

**Problem:** On macOS ARM, the COMGR dylib (`libamd_comgr.dylib`) and Metal's shader compiler share LLVM library state. If Metal initializes its LLVM runtime first (which happens when `Transformer(config)` creates the hundreds of `Tensor.zeros()` weight tensors targeting the METAL device), loading the COMGR dylib afterward causes a SIGBUS in `amd_comgr_do_action`.

**Fix:** `_comgr_prewarm()` is called at the very start of `from_safetensors`, before the safetensors header is read and before `Transformer(config)`. It compiles a trivial dummy HIP kernel, which loads the COMGR dylib and initializes its LLVM state first.

**Critical invariant:** `_comgr_prewarm()` must always be called BEFORE `Transformer(config)`. Moving it even one step later (e.g., after header reading, after model creation) causes the crash to return.

### 2. AMD block dispatch without precompile

`FFNBlock.__call__` uses `@function(precompile=True, allow_implicit=True)` which compiles the entire block (attention + FFN) as one program. For 27B model blocks, this shader is too large and causes COMGR to fail.

For AMD blocks, `_patch_for_multidevice` overrides dispatch via `_amd_block_call`, which wraps the block computation in `@function(precompile=False, allow_implicit=True)`. This defers compilation to the outer `TinyJit`, which compiles individual kernels.

Note: `block.__call__ = ...` instance patching doesn't work because Python resolves `__call__` on the class. The fix routes through a function dispatched in `_patched_forward`.

### 3. Device boundary management

`_patched_forward` inserts `.to(device)` calls when the activation tensor moves between devices (Metal → AMD or AMD → Metal). Non-block tensors (`token_embd`, `output`, `output_norm`) live on Metal.

### 4. freqs_cis device patch

`TransformerBlock._init_state` creates `freqs_cis` on the default device (Metal). For AMD blocks, `_make_patched_init` patches `_init_state` to move `freqs_cis` to the block's device before the JIT captures it.

## Testing

```bash
# Default (AMD auto-placement, up to 20 GB):
python extra/quantize_qwen_smoke.py ~/hug-models/Qwen3.6-27B/qwen3.6-27b-int8-unified.safetensors

# Force all-Metal (useful if AMD is unavailable or unstable):
python extra/quantize_qwen_smoke.py ~/hug-models/Qwen3.6-27B/qwen3.6-27b-int8-unified.safetensors --amd-budget-gb 0

# Fresh kernel cache (bypasses SQLite cache, forces full recompilation):
CACHEDB=/tmp/fresh.db python extra/quantize_qwen_smoke.py ...
```

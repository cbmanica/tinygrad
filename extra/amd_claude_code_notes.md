# AMD RX 7900 XTX as Claude Code Backend — What We Learned

## Goal

Use the AMD RX 7900 XTX (24 GB VRAM, Sonnet Breakaway Box, USB4/Thunderbolt) to serve a local LLM as a Claude Code backend, replacing Ollama+Qwen3.6-27B running on Metal.

---

## Attempt 1: AMD+Metal Split Inference on Qwen3.6-27B int8

**Branch:** `amd-qwen-split-v2`

**What we did:** Distributed 64 layers across AMD (52 layers, ~19.76 GB) and Metal (12 layers). Required:
- `_comgr_prewarm()` — compile a dummy HIP kernel before any Metal tensors are created, to prevent SIGBUS from COMGR/Metal LLVM state conflict on macOS ARM
- `precompile=False` for AMD blocks (`_amd_block_call`) to defer to TinyJit instead of the per-block `@function(precompile=True)` which produced oversized shaders
- `freqs_cis` device patching for full-attention TransformerBlocks assigned to AMD
- `_patched_forward` to insert `.to(device)` at AMD↔Metal boundaries

**Benchmark result:** AMD+Metal split was ~10% *slower* than Metal-only.

**Why it failed:** On macOS, tinygrad accesses the RX 7900 XTX via TinyGPU.app (IOKit / APLRemotePCIDevice), which proxies every kernel dispatch over a Unix socket (`/tmp/tinygpu.sock`). Each kernel is a full round-trip. Metal, by contrast, batches the entire forward pass into a single command buffer. The per-kernel socket latency completely dominates over any memory bandwidth advantage from the AMD card. Thunderbolt 5 would not help — the bottleneck is dispatch latency, not bandwidth.

**Also:** Auto-placement puts blocks 0–47 on AMD, but blocks 48–63 alternate AMD/Metal (full-attn at 51,55,59,63 on AMD; SSM blocks on Metal), causing 8+ cross-device transfers per token in the model's tail.

---

## Attempt 2: All-AMD Inference via GGUF

**Branch:** `gguf-amd-device`

**What we added to cli.py:**
- `--device DEV` flag: sets `DEV.value = dev` before model load; for AMD, also runs COMGR prewarm (`compile_hip` with `__attribute__((global))` kernel — note: `__global__` is *not* valid here because `compile_hip` uses `-nogpuinc`)
- `--serve` now binds to `127.0.0.1` only (was `''`, all interfaces)
- `qwen3.5:35b-a3b-ks` model entry (Q4_K_S, 20.7 GB, fits in 24 GB AMD VRAM)
- POST `/v1/messages` handler (Anthropic format) — see below

### Model: Qwen3.5-35B-A3B-Q4_K_S

**Why chosen:** 35B total / 3B active MoE, 20.7 GB at Q4_K_S — fits in 24 GB. Originally believed to be pure-attention MoE.

**What we discovered:** Qwen3.5-35B-A3B has **GatedDeltaNet (recurrent/SSM) blocks**. In tinygrad, `GatedDeltaNetBlock` has `assert T == 1` — cannot process more than one token at a time. This forces `has_recurrent_block = True` → `chunk_size = 1` in `generate()`. Prefill is entirely sequential. For a 4000-token Claude Code context: 4000 × 43 ms = **~2.9 minutes per turn**. Unusable.

The "3.5" Qwen generation introduced a DeltaNet hybrid architecture. We were wrong to assume it was pure attention. The distinction:
- `qwen3:30b-a3b` — Qwen3 generation, standard attention + MoE, no recurrent blocks ✓
- `qwen3.5:35b-a3b` — Qwen3.5 generation, GatedDeltaNet + MoE, chunk_size forced to 1 ✗
- `qwen3.6:35b-a3b` — Qwen3.6 generation, not supported by tinygrad GGUF loader at all ✗

### Model: Qwen3-30B-A3B-Q4_K_M

**Result:** `has_recurrent_block = False`, `chunk_size = 32`. Model ~18.5 GB. Decode throughput: **23 tok/s at ~80 GB/s**. Good.

**Prefill problem:** Claude Code sends ~4000 tokens per turn. Logging showed `in: 15 +3931` on every turn — only 15 tokens of KV cache reused. Claude Code's system prompt includes dynamic content (timestamp or similar) that changes after token 15, invalidating the rest of the KV cache on every message.

With `chunk_size = 32`, 3931 tokens → 123 chunks. Expected speedup; actual result: **still ~3 minutes**, same as chunk_size=1. Per-chunk time: ~1.44 s = exactly 32 × 43 ms.

**Why chunk_size=32 gave no speedup:** Attention reads `cache[:, :, :, 0:start_pos+32, :]` where `start_pos` grows by 32 each chunk. Each new `start_pos` value produces a different-shaped attention matrix. Tinygrad recompiles a new HIP kernel for each shape — ~123 compilations instead of 1. This is a tinygrad symbolic-shape limitation for AMD: the attention context dimension is not handled as a truly dynamic shape through the JIT.

**What would fix it:** The TinyJit needs to compile one attention kernel that handles variable `start_pos` without recompilation. Until that's implemented, AMD prefill for long contexts will be as slow as sequential decode.

---

## Proxy Setup: Anthropic Messages API

Claude Code sends POST `/v1/messages` (Anthropic format). Tinygrad's server only had `/v1/chat/completions` (OpenAI format).

**LiteLLM attempt:** Failed. Newer LiteLLM routes to the OpenAI Responses API (`/responses` or `/v1/responses`) instead of chat completions. No combination of `use_responses_api: false`, `force_chat_completions: true`, or provider prefixes reliably prevented this.

**Solution:** Implemented `/v1/messages` directly in `Handler.do_POST` in `tinygrad/llm/cli.py`:
- `_content_to_text()`: text → tokenize; tool_use → `<tool_use name="...">...</tool_use>` XML; tool_result → similar; thinking/redacted_thinking → skip
- `_stream_anthropic()`: proper Anthropic SSE sequence: `message_start` → `content_block_start` → `ping` → `content_block_delta` × N → `content_block_stop` → `message_delta` → `message_stop`
- System prompt extracted from top-level `system` field (str or list of text blocks)
- Non-streaming path also implemented
- Path matched as `self.path.split("?")[0] == "/v1/messages"` to handle `?beta=true` query string

**Caddy notes:**
- `brew services restart caddy` does NOT restart the running Caddy process — it starts a new process that immediately crashes (looks for `/opt/homebrew/etc/Caddyfile`). Use `caddy reload --config /usr/local/etc/caddy/Caddyfile` to reload the live process.
- `@authenticated { header Authorization "*bBsfyajs?ay8D&y!*" }` — the `?` is a glob wildcard in Caddy, but still correctly matches a literal `?` in the token value.
- 502 from Caddy means auth passed and the upstream failed (connection refused or exception). `abort` (auth failure) shows differently. `bytes_read: 0` with a 502 can mean the upstream threw an unhandled exception and closed the connection before sending a response.

---

## Final Working Setup

**Caddy** (`/usr/local/etc/caddy/Caddyfile`, reload with `caddy reload --config ...`):
- Port **8080** → Ollama at `127.0.0.1:11434` (Qwen3.6-27B int4, fast Metal prefill)
- Port **8081** → tinygrad at `127.0.0.1:8000` (Qwen3-30B-A3B on AMD, 23 tok/s decode, slow prefill)

**Switching in Claude Code** (remote machine):
```bash
export ANTHROPIC_BASE_URL=http://zeal:8080   # Ollama / Metal (recommended for now)
export ANTHROPIC_BASE_URL=http://zeal:8081   # tinygrad / AMD
```

**Starting tinygrad AMD server:**
```bash
source /Users/bridget/dev/tinygrad/venv/bin/activate
python -m tinygrad.llm.cli --model qwen3:30b-a3b --device AMD --max_context 32768 --serve
```

Logs: `~/Library/Logs/caddy_ollama.log`, `~/Library/Logs/caddy_amd.log`.

---

## What Would Need to Change for AMD to Be Competitive

1. **Symbolic attention dispatch in TinyJit:** Compile one kernel per attention pattern that handles variable `start_pos` without recompilation. This is the main blocker for usable prefill on long contexts.

2. **Parallel scan for GatedDeltaNet:** Would enable `chunk_size > 1` for Qwen3.5/3.6 hybrid models, making large-context prefill viable for those architectures.

3. **TinyGPU kernel batching:** Batching multiple kernel dispatches per socket message would reduce the per-kernel latency overhead that makes split-device inference uncompetitive vs. Metal command buffers.

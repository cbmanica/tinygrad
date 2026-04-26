This reanalysis provides a path forward to rectify the issues encountered during the Qwen3.6-27B integration into Tinygrad. It addresses the likely failure points in the initial GGUF-to-Safetensors conversion and outlines a robust, multi-device offloading strategy.

***

# reanalysis_plan.md

## 1. Retrospective: The "Zero-Value" and "Renderer" Root Causes

Our analysis of the `model.py` execution history and the issues with `qwen3.6-27b-int8.safetensors` reveals a fundamental mismatch in the initial conversion process.

* **The Quantization Mismatch**: The current `int8.safetensors` was derived from a `Q4_K_M` GGUF. GGUF uses "block-based" or "super-block" quantization where weights are stored in small clusters (e.g., 32 weights) with their own shared scales and mins. If the conversion to Safetensors simply dumped these raw buffers or used a generic `int8` cast without deconstructing the GGUF super-blocks, the resulting weights in Tinygrad appear as "zeros" or noise because the essential block-level scaling factors were lost or incorrectly mapped.
* **The Renderer Error**: The `NotImplementedError: needs a renderer` occurred because we attempted to perform a `cast()` or `realize()` operation on a tensor still residing on the `DISK` device. In Tinygrad, the `DISK` device is a pass-through to a file descriptor; it has no compute engine to perform even a basic data-type cast.
* **The Architecture Pivot**: We started with Qwen3.5-35B-A3B (an MoE model) but transitioned to Qwen3.6-27B (a Dense Hybrid model). These are fundamentally different architectures. The 3.6 model's unique 3:1 layer ratio (DeltaNet vs. Attention) requires a custom implementation that our initial GGUF source may not have fully supported or described.

---

## 2. Proposed "From Scratch" Pipeline

To ensure model integrity, we should move away from GGUF-to-Safetensors scripts and utilize a native Tinygrad loading process.

### Phase 1: Clean Weight Acquisition
1.  **Source**: Download the **Official BF16 Safetensors** from the [Qwen3.6-27B Hugging Face repo](https://huggingface.co/Qwen/Qwen3.6-27B).
2.  **Why?**: These weights are unquantized and follow a standard format. This eliminates the "gibberish" results often seen when converting between lossy formats like GGUF.

### Phase 2: Native Tinygrad Quantization
Instead of loading an external `int8` file, we will perform a **Native Loader Quantization**:
1.  Load the BF16 tensor from disk.
2.  Move it to RAM (`CPU`).
3.  Calculate the `scale = max(abs(v)) / 127`.
4.  Quantize to `int8` (`char`) and store the `scale` as a separate tensor.
5.  **Save**: Export this as a `qwen3.6-27b-tinygrad.safetensors` file where each weight tensor has a corresponding `_scale` tensor.

---

## 3. Architecture Implementation: The 3:1 Hybrid Scheme

Qwen3.6-27B has **64 layers** organized into 16 blocks of 4. We must implement the `Transformer` class to strictly follow this repeating pattern:

* **Sub-layers 0, 1, 2**: `Gated DeltaNet` (Linear Attention with 48 V-heads, 16 QK-heads).
* **Sub-layer 3**: `Gated Attention` (Standard Global Attention with 24 Q-heads, 4 KV-heads).
* **FFN**: Every sub-layer is followed by a Gated FFN (Intermediate Dim: 17,408).

---

## 4. Configurable Hybrid Offloading Strategy

To split the 64 layers between an **AMD GPU** and a **Mac CPU/GPU (Metal)**, we will implement a `DeviceMap` configuration.

### Implementation Logic:
```python
@dataclass
class DeviceMap:
    primary_gpu: str = "AMD"   # External GPU
    secondary_gpu: str = "METAL" # Mac Integrated GPU
    offload_start_layer: int = 32 # Configurable split point

# Inside Transformer __init__
for i in range(64):
    target_device = map.primary_gpu if i < map.offload_start_layer else map.secondary_gpu
    # Assign the layer to the specific device immediately during loading
    self.layers[i].to(target_device)
```

### Why this works:
1.  **Memory Balancing**: The Gated Attention layers (every 4th layer) require the most KV cache memory. We can strategically place these on the device with higher VRAM.
2.  **Compute Splitting**: DeltaNet layers are highly efficient and can be offloaded to the secondary device (Metal/CPU) without significant bottlenecking, while keeping the standard Attention layers on the high-performance AMD card.

---

## 5. Immediate Action Items

1.  **File Cleanup**: Discard the current `qwen3.6-27b-int8.safetensors`. It is the source of the zero-weight artifacts.
2.  **Official HF Fetch**: Download the `model.safetensors.index.json` and the corresponding `.safetensors` shards from the official Qwen/Qwen3.6-27B repository.
3.  **Loader Update**: Refactor the `load_state_dict` in `model.py` to strip prefixes *before* attempting any device moves, and implement the `_scale` multiplication logic natively.
4.  **Device Targeting**: Use `Device["AMD"]` and `Device["METAL"]` explicitly. For Apple Silicon, Metal is significantly faster than CPU and supports unified memory, which is ideal for the 256K context window.

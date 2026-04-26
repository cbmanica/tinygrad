import os, pathlib, json, sys, gc
import numpy as np
from tinygrad import Tensor, dtypes, Context
from tinygrad.helpers import DEV
from tinygrad.nn.state import safe_load, safe_save

# Force the process to stay off the AMD card
DEV.value = "METAL"

if len(sys.argv) < 2 or sys.argv[1] not in ["4", '6', "8"]:
    print("Usage: python3 quantize.py [4|8]")
    sys.exit(1)

BITS = int(sys.argv[1])
model_path = pathlib.Path("~/hug-models/Qwen3.6-27B/qwen3.6-27b-f16.safetensors").expanduser()
out_path = model_path.parent / f"qwen3.6-27b-int{BITS}.safetensors"

print(f"Loading 16-bit weights for {BITS}-bit quantization on METAL/System RAM...")

with Context(DEV="METAL"):
    state_dict = safe_load(str(model_path))
    quantized_dict = {}

    for k, v in state_dict.items():
        # Only quantize the heavy hitters (2D matrices)
        if len(v.shape) > 1:
            print(f"  Quantizing {k}...")
            # bitcast + numpy to bypass the broken BF16 CPU/Metal renderer
            # We treat bits as int16, then convert to f32 in NumPy to do the scaling
            raw_bits = v.bitcast(dtypes.int16).numpy()
            
            # Manual BF16 -> F32 bit-shift
            f32_bits = raw_bits.ravel().astype(np.uint32) << 16
            arr = f32_bits.view(np.float32)
            
            if BITS == 8:
                scale = np.abs(arr).max() / 127.0
                q_arr = (arr / (scale + 1e-8)).round().astype(np.int8)
                quantized_dict[k] = Tensor(q_arr.reshape(v.shape))
            
            elif BITS == 4:
                # Simple packing: 2 weights per byte
                scale = np.abs(arr).max() / 7.0
                q_arr = (arr / (scale + 1e-8)).round().clip(-8, 7).astype(np.int8)
                
                # We need the last dimension to be even for packing
                flat = q_arr.flatten()
                # Ensure we have an even number of elements
                if flat.size % 2 != 0:
                    flat = np.append(flat, 0)
                    
                # Pack two 4-bit values into one int8 byte
                packed = ((flat[::2] & 0x0F) | (flat[1::2] << 4)).astype(np.int8)
                new_shape = (v.shape[0], v.shape[1] // 2)
                quantized_dict[k] = Tensor(packed.reshape(new_shape))
        else:
            # Keep small tensors (norms, biases) as-is
            quantized_dict[k] = v

        # Periodic cleanup of the intermediate NumPy arrays
        if len(quantized_dict) % 10 == 0:
            gc.collect()

    print(f"Saving {BITS}-bit model to {out_path}...")
    safe_save(quantized_dict, str(out_path))

print("Done. AMD VRAM remains untouched.")

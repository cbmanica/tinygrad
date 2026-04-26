import os, pathlib, json, sys, gc
import numpy as np
from tinygrad import Tensor, dtypes, Context
from tinygrad.helpers import DEV
from tinygrad.nn.state import safe_load, safe_save

DEV.value = "METAL" # Use the Mac Mini's RAM

model_path = pathlib.Path("~/hug-models/Qwen3.6-27B/qwen3.6-27b-f16.safetensors").expanduser()
out_path = model_path.parent / "qwen3.6-27b-int6.safetensors"

print("Quantizing to 6-bit (Stored in INT8 containers)...")

with Context(DEV="METAL"):
    state_dict = safe_load(str(model_path))
    quantized_dict = {}

    for k, v in state_dict.items():
        if len(v.shape) == 2 and v.numel() > 1024:
            print(f"  Processing {k}...")
            # Bypassing the broken BF16 renderer
            raw_bits = v.bitcast(dtypes.int16).numpy()
            f32_bits = raw_bits.ravel().astype(np.uint32) << 16
            arr = f32_bits.view(np.float32)
            
            # Scale to 6-bit range (-32 to 31)
            scale = np.abs(arr).max() / 31.0
            # We store it in a standard int8 so tinygrad can 'see' it without a custom C++ kernel
            q_arr = (arr / (scale + 1e-8)).round().clip(-32, 31).astype(np.int8)
            
            quantized_dict[k] = Tensor(q_arr.reshape(v.shape))
            quantized_dict[f"{k}.scale"] = Tensor([scale], dtype=dtypes.float16)
        else:
            quantized_dict[k] = v

    print(f"Saving 6-bit-in-8-bit model to {out_path}...")
    safe_save(quantized_dict, str(out_path))

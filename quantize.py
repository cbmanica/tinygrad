# This was the mechanism used to quantize the qwen model for 8bit
import os
import pathlib
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load, safe_save

# 1. Load into System RAM
model_path = pathlib.Path("~/hug-models/Qwen3.6-27B/qwen3.6-27b-flat.safetensors").expanduser()
print("Loading 55GB weights...")
state_dict = safe_load(str(model_path))

quantized_dict = {}
for k, v in state_dict.items():
    print(f"Processing {k}...")
    # .cast(dtypes.half) changes the type
    # .realize() forces the computation to happen (BF16 -> F16)
    # .to("CPU") ensures it stays in your 64GB System RAM and doesn't touch the GPU yet
    quantized_dict[k] = v.cast(dtypes.half).realize().to("CPU")

# 2. Save the 27GB version to Zeal
out_path = model_path.parent / "qwen3.6-27b-f16.safetensors"
print(f"Saving to {out_path}...")
safe_save(quantized_dict, str(out_path))
print("Done.")

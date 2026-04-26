# This is the method used to create a single safetensors output for the qwen model.
import os, pathlib, json, gc
import numpy as np
from tinygrad import Tensor, dtypes, Context
from tinygrad.helpers import DEV
from tinygrad.nn.state import safe_load, safe_save

# 1. Target the Mac's internal GPU for the consolidation
DEV.value = "METAL"

model_dir = pathlib.Path("~/hug-models/Qwen3.6-27B").expanduser()
index_path = model_dir / "model.safetensors.index.json"
out_path = model_dir / "qwen3.6-27b-f16.safetensors"

with open(index_path, "r") as f:
    weight_map = json.load(f)["weight_map"]

shards = sorted(list(set(weight_map.values())))
final_state = {}

# 2. Use the Context manager for the internal GPU
with Context(DEV="METAL"):
    for shard in shards:
        print(f"Loading shard {shard}...")
        shard_state = safe_load(str(model_dir / shard))
        
        for k, v in shard_state.items():
            print(f"  Converting {k}...")
            # bitcast bypasses the renderer errors
            raw_bits = v.bitcast(dtypes.int16).numpy()
            
            # NumPy conversion in System RAM
            f32_bits = raw_bits.ravel().astype(np.uint32) << 16
            f16_data = f32_bits.view(np.float32).astype(np.float16).reshape(v.shape)
            
            # Wrap as a Metal-backed tensor (stored in unified memory)
            final_state[k] = Tensor(f16_data)
        
        del shard_state
        gc.collect()

    print(f"Saving 27GB consolidated file to {out_path}...")
    safe_save(final_state, str(out_path))

print("Success. Consolidated using M4 Pro internal GPU.")

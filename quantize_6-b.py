import numpy as np
from tinygrad import Tensor, dtypes, Context
from tinygrad.nn.state import safe_load, safe_save

# This script actually PACKS the bits to save space
with Context(DEV="METAL"):
    state_dict = safe_load("qwen3.6-27b-f16.safetensors")
    packed_dict = {}

    for k, v in state_dict.items():
        if len(v.shape) == 2 and v.shape[1] % 4 == 0:
            print(f"Packing {k}...")
            # 1. Scale weights to 6-bit range (0-63)
            arr = v.numpy().astype(np.float32)
            scale = np.abs(arr).max() / 63.0
            q_arr = (arr / (scale + 1e-8)).round().clip(0, 63).astype(np.uint8)
            
            # 2. PACKING: 4 weights -> 3 bytes
            w = q_arr.reshape(-1, 4)
            b0 = (w[:, 0] << 2) | (w[:, 1] >> 4)
            b1 = ((w[:, 1] & 0x0F) << 4) | (w[:, 2] >> 2)
            b2 = ((w[:, 2] & 0x03) << 6) | w[:, 3]
            
            # 3. Store as a flat UINT8 tensor
            packed_bytes = np.stack([b0, b1, b2], axis=1).flatten()
            packed_dict[k] = Tensor(packed_bytes, dtype=dtypes.uint8)
            packed_dict[f"{k}.scale"] = Tensor([scale], dtype=dtypes.float16)
        else:
            packed_dict[k] = v

    safe_save(packed_dict, "qwen3.6-27b-int6_packed.safetensors")

import os, json
from pathlib import Path
from tinygrad import Tensor, nn, dtypes, Device
from tinygrad.helpers import tqdm

MODEL_DIR = Path("~/hug-models/Qwen3.6-27B").expanduser()
OUTPUT_FILE = MODEL_DIR / "qwen3.6-27b-int8-unified.safetensors"

def quantize_to_int8(t: Tensor):
    # Symmetric quantization: scale = max(abs(v)) / 127
    v_max = t.abs().max().numpy()
    scale = v_max / 127.0
    # Bridge to CPU for the cast to avoid "needs a renderer" error
    t_quant = (t / scale).round().cast(dtypes.char).to("CPU").realize()
    return t_quant, Tensor(scale, dtype=dtypes.float32)

# 1. Map shards from index
with open(MODEL_DIR / "model.safetensors.index.json", "r") as f:
    index = json.load(f)

full_state_dict = {}
shard_files = sorted(list(set(index["weight_map"].values())))

print(f"Loading {len(shard_files)} shards...")
for shard in tqdm(shard_files):
    sd = nn.state.safe_load(MODEL_DIR / shard)
    for k, v in sd.items():
        # Clean the key name immediately
        clean_k = k.replace("model.language_model.", "").replace("model.", "")
        if clean_k == "embed_tokens.weight": clean_k = "token_embd.weight"
        elif clean_k == "norm.weight": clean_k = "output_norm.weight"
        elif clean_k == "lm_head.weight": clean_k = "output.weight"
        
        # Move to CPU to bridge the DISK device
        full_state_dict[clean_k] = v.to("CPU").realize()

# 2. Quantize and build final dict
final_sd = {}
print("Quantizing weights...")
for k, v in tqdm(full_state_dict.items()):
    if "weight" in k and v.ndim > 1 and "token_embd" not in k:
        q_weight, scale = quantize_to_int8(v)
        final_sd[k] = q_weight
        final_sd[f"{k}_scale"] = scale
    else:
        # Keep biases and embeddings in full precision (recommended)
        final_sd[k] = v

# 3. Save as single unified file
print(f"Saving to {OUTPUT_FILE}...")
nn.state.safe_save(final_sd, str(OUTPUT_FILE))

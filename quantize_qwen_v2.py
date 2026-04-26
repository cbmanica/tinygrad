import os, json
from pathlib import Path
from tinygrad import Tensor, nn, dtypes, Device
from tinygrad.helpers import tqdm

MODEL_DIR = Path("~/hug-models/Qwen3.6-27B").expanduser()
OUTPUT_FILE = MODEL_DIR / "qwen3.6-27b-int8-unified.safetensors"

def quantize_to_int8(t: Tensor):
    # STEP 1: Move raw data to CPU and realize to avoid the 'renderer' error
    t_cpu = t.to("CPU").realize()
    
    # STEP 2: Now we can safely perform math on the CPU-backed tensor
    v_max = t_cpu.abs().max().numpy().item()
    scale = float(v_max) / 127.0 if v_max > 0 else 1.0
    
    # STEP 3: Quantize and cast
    # We use (t_cpu / scale) because t_cpu is already realized in RAM
    t_quant = (t_cpu / scale).round().cast(dtypes.char).realize()
    
    return t_quant, Tensor(scale, dtype=dtypes.float32)

# 1. Map shards from index
index_path = MODEL_DIR / "model.safetensors.index.json"
if not index_path.exists():
    raise FileNotFoundError(f"Missing index file at {index_path}")

with open(index_path, "r") as f:
    index = json.load(f)

final_sd = {}
shard_files = sorted(list(set(index["weight_map"].values())))

print(f"Processing {len(shard_files)} shards...")
for shard in tqdm(shard_files):
    # Load shard directly into memory-mapped state
    sd = nn.state.safe_load(MODEL_DIR / shard)
    
    for k, v in sd.items():
        # Clean the key names for model.py compatibility
        clean_k = k.replace("model.language_model.", "").replace("model.", "")
        if clean_k == "embed_tokens.weight": clean_k = "token_embd.weight"
        elif clean_k == "norm.weight": clean_k = "output_norm.weight"
        elif clean_k == "lm_head.weight": clean_k = "output.weight"
        
        # Quantize weight matrices (excluding embeddings and 1D vectors like biases/norms)
        if "weight" in clean_k and v.ndim > 1 and "token_embd" not in clean_k:
            q_weight, scale = quantize_to_int8(v)
            final_sd[clean_k] = q_weight
            final_sd[f"{clean_k}_scale"] = scale
        else:
            # Move non-quantized layers to CPU memory to decouple from the shard file
            final_sd[clean_k] = v.to("CPU").realize()
            
    # Explicitly clear the shard state-dict to free up RAM before next shard
    del sd 

# 3. Save as a single, consolidated 8-bit file
print(f"Saving to {OUTPUT_FILE}...")
nn.state.safe_save(final_sd, str(OUTPUT_FILE))
print("Quantization complete.")

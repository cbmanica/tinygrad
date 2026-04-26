import json, pathlib
from tinygrad.nn.state import safe_load, safe_save

# Using paths for Zeal
model_dir = pathlib.Path("~/hug-models/Qwen3.6-27B").expanduser()
index_path = model_dir / "model.safetensors.index.json"

with open(index_path, "r") as f:
    weight_map = json.load(f)["weight_map"]

# Get unique shard filenames
shards = sorted(list(set(weight_map.values())))

state_dict = {}
for shard in shards:
    print(f"Loading {shard}...")
    # This works because each shard is a valid standalone safetensor file
    state_dict.update(safe_load(str(model_dir / shard)))

out_file = model_dir / "qwen3.6-27b-flat.safetensors"
print(f"Saving to {out_file}...")
safe_save(state_dict, str(out_file))

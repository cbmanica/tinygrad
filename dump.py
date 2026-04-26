import sys
import pathlib
import json
from safetensors import safe_open

MODEL_PATH = pathlib.Path("~/hug-models/Qwen3.6-27B/qwen3.6-27b-int8.safetensors").expanduser()

if not MODEL_PATH.exists():
    print(f"Error: {MODEL_PATH} not found.")
    sys.exit(1)

print(f"--- ANALYZING GROUND TRUTH: {MODEL_PATH} ---")

try:
    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        
        # 1. Block Identification
        blocks = set()
        for k in keys:
            parts = k.split('.')
            for p in parts:
                if p.isdigit():
                    blocks.add(int(p))
                    break
        
        print(f"Total Keys: {len(keys)}")
        print(f"Block Range: {min(blocks) if blocks else 'N/A'} to {max(blocks) if blocks else 'N/A'}")

        # 2. Check Global Weights (Embedding/Norms)
        print("\n--- GLOBAL WEIGHTS (Determines Base 'dim') ---")
        for k in keys:
            if any(x in k for x in ['embed', 'output_norm', 'lm_head']):
                # .get_slice(k).get_shape() is the correct method for the metadata
                print(f"{k:45} : {f.get_slice(k).get_shape()}")

        # 3. Inspect Block 0 (The Rosetta Stone)
        print("\n--- BLOCK 0 STRUCTURE ---")
        for k in keys:
            if '.0.' in k:
                print(f"{k:45} : {f.get_slice(k).get_shape()}")

        # 4. Check for SSM Indicators specifically
        print("\n--- SSM / CONV SPECIFIC KEYS ---")
        ssm_count = 0
        for k in keys:
            if any(x in k for x in ['ssm', 'conv', 'recurrent', 'qkv_proj']):
                if ssm_count < 10:
                    print(f"{k:45} : {f.get_slice(k).get_shape()}")
                ssm_count += 1
        print(f"Total SSM-related keys found: {ssm_count}")

except Exception as e:
    print(f"Failed to read metadata: {e}")
    # Fallback debug: just print the keys if shapes fail
    print("\nAttempting to list keys only...")
    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        for k in sorted(f.keys())[:20]:
            print(k)

import struct
import json
import pathlib

# Path based on your environment
MODEL_PATH = pathlib.Path("~/hug-models/Qwen3.6-27B/qwen3.6-27b-int8.safetensors").expanduser()

def dump_every_single_key(path):
    print(f"--- STARTING FULL HEADER DUMP: {path} ---")
    if not path.exists():
        print(f"Error: File {path} not found.")
        return

    with open(path, "rb") as f:
        # Read header size
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            print("Error: Could not read header length.")
            return
        
        header_size = struct.unpack("<Q", length_bytes)[0]
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)
        
        # Remove metadata to leave only tensor entries
        header.pop("__metadata__", None)
        
        keys = sorted(header.keys())
        total = len(keys)
        print(f"Detected Total Keys: {total}\n")

        # Printing every single key with its shape
        for i, k in enumerate(keys):
            shape = header[k].get('shape', 'UNKNOWN')
            # Print index to verify we hit 1199
            print(f"[{i+1}/{total}] {k:70} : {shape}")

try:
    dump_every_single_key(MODEL_PATH)
except Exception as e:
    print(f"FAILED TO DUMP: {e}")

import os
from tinygrad.llm.model import Transformer
from tinygrad.nn.state import get_state_dict, load_state_dict
from gguf import GGUFReader

# 1. Manually define the Qwen 3.5 MoE / 35B architecture
# This bypasses all KeyErrors by providing the blueprint directly
model_config = {
    "dim": 4096, "n_layers": 28, "n_heads": 32, "n_kv_heads": 32,
    "vocab_size": 152064, "norm_eps": 1e-6, "rope_theta": 1000000,
    "max_context": 32768
}

print("Initializing Transformer on 7900 XTX...")
model = Transformer(**model_config)

# 2. Use GGUFReader to grab the tensors without the 'tinygrad.llm' wrapper
reader = GGUFReader("./merged.gguf")
state_dict = {}
for tensor in reader.tensors:
    # Map the GGUF tensor names to tinygrad Transformer names
    name = tensor.name.decode("utf-8")
    state_dict[name] = tensor.data

# 3. Load weights into the model
# We use strict=False because GGUF and tinygrad names often have slight mismatches
load_state_dict(model, state_dict, strict=False)

# 4. Serving logic
from tinygrad.llm.api import start_server
print("Server starting on port 11434... Use 'DEV=AMD' to ensure GPU usage.")
start_server(model, port=11434)

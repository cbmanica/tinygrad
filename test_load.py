# test_load.py
# Smoke test before trying to actually run the model
from tinygrad.llm.model import Transformer, TransformerConfig
from pathlib import Path

path = Path("~/hug-models/Qwen3.6-27B/qwen3.6-27b-int8-unified.safetensors").expanduser()

# TARGETED FIX: Reduce offload_layers to fit in 24GB VRAM.
# Try setting this to 32 (half the 64 layers) to balance AMD and Metal/CPU.
config = TransformerConfig(offload_layers=32) 

model = Transformer.load_quantized(path, config)

# Smoke test
from tinygrad import Tensor
test_input = Tensor([[1, 2, 3]])
logits = model(test_input, 0)

print(f"Logits shape: {logits.shape}")
print(f"Logits sample (first 5): {logits.numpy()[0, 0, :5]}")

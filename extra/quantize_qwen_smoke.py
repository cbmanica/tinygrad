"""Quick smoke test: load safetensors model + tokenizer, run 5-token generation.

Usage:
  python extra/quantize_qwen_smoke.py path/to/model.safetensors [--amd-budget-gb N]

Default is --amd-budget-gb 20 (auto-placement with up to 20 GB on AMD).
Pass --amd-budget-gb 0 to force all-METAL.
"""
import sys, pathlib, argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("path", type=str, help="Path to unified .safetensors file")
  parser.add_argument("--amd-budget-gb", type=float, default=20.0,
                      help="AMD VRAM budget in GB (0=all-METAL, default 20)")
  args = parser.parse_args()

  path = pathlib.Path(args.path).expanduser()
  assert path.exists(), f"File not found: {path}"

  from tinygrad.llm.hf_tokenizer import from_hf_tokenizer_json
  from tinygrad.llm.safetensors_loader import from_safetensors

  print("Loading tokenizer...")
  tok = from_hf_tokenizer_json(path.parent)
  ids = tok.encode("Hello, world.")
  roundtrip = tok.decode(ids)
  print(f"Tokenizer OK: {ids!r} → {roundtrip!r}")

  print("Loading model...")
  model = from_safetensors(path, max_context=256, amd_budget_gb=args.amd_budget_gb)
  print("Model loaded.")

  prompt = tok.prefix() + tok.role("user") + tok.encode("Say hi in one word.") + tok.end_turn() + tok.role("assistant")
  print(f"Prompt tokens: {len(prompt)}")

  print("Generating 5 tokens:")
  out = []
  dec = tok.stream_decoder()
  for token_id in model.generate(prompt):
    out.append(token_id)
    sys.stdout.write(dec(token_id))
    sys.stdout.flush()
    if tok.is_end(token_id) or len(out) >= 5:
      break
  sys.stdout.write(dec() + "\n")
  print(f"Generated {len(out)} tokens: {out}")
  print("SMOKE TEST PASSED")

if __name__ == "__main__":
  main()

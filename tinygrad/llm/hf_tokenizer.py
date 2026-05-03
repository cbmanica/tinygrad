from __future__ import annotations
import json, pathlib
from tinygrad.llm.cli import SimpleTokenizer


def from_hf_tokenizer_json(model_dir: str | pathlib.Path) -> SimpleTokenizer:
  """Load a SimpleTokenizer from a Hugging Face model directory containing tokenizer.json."""
  model_dir = pathlib.Path(model_dir)

  with open(model_dir / "tokenizer.json") as f:
    tok_data = json.load(f)

  normal_tokens: dict[str, int] = tok_data["model"]["vocab"]
  special_tokens: dict[str, int] = {t["content"]: t["id"] for t in tok_data["added_tokens"]}

  bos_id: int | None = None
  eos_id: int = 0
  eot_id: int | None = None

  # Priority: generation_config.json > tokenizer_config.json > config.json
  for cfg_name in ["generation_config.json", "tokenizer_config.json", "config.json"]:
    cfg_path = model_dir / cfg_name
    if not cfg_path.exists():
      continue
    try:
      with open(cfg_path) as f:
        cfg = json.load(f)
      # eos
      eos_raw = cfg.get("eos_token_id")
      if eos_raw is not None:
        eos_id = eos_raw[0] if isinstance(eos_raw, list) else int(eos_raw)
        eot_id = eos_id
      # bos — only set if add_bos_token is True
      if cfg.get("add_bos_token", False):
        bos_raw = cfg.get("bos_token_id")
        if bos_raw is not None:
          bos_id = int(bos_raw)
      if eos_raw is not None:
        break
    except Exception:
      continue

  return SimpleTokenizer(normal_tokens, special_tokens, preset="qwen2",
                         bos_id=bos_id, eos_id=eos_id, eot_id=eot_id)

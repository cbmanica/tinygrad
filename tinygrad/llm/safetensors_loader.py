from __future__ import annotations
import struct, json, pathlib
from tinygrad import Tensor, nn
from tinygrad.engine.jit import TinyJit
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig
from tinygrad.llm.quant import replace_linears_with_quantized
from tqdm import tqdm


def _read_header(path: pathlib.Path) -> tuple[dict, dict]:
  with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
  return header, header.get("__metadata__", {})


def _device_for_key(key: str, placement: list[str]) -> str:
  # Non-block tensors (token_embd, output, output_norm) go to METAL
  parts = key.split(".")
  if parts[0] == "blk" and len(parts) > 1 and parts[1].isdigit():
    i = int(parts[1])
    if i < len(placement):
      return placement[i]
  return "METAL"


def _resolve_placement(config: TransformerConfig, amd_layers, metal_layers, cpu_layers,
                       amd_budget_gb: float, block_sizes: dict[int, int]) -> list[str]:
  num_blocks = config.num_blocks

  if any(x is not None for x in [amd_layers, metal_layers, cpu_layers]):
    placement = ["METAL"] * num_blocks
    if amd_layers:
      for i in amd_layers: placement[i] = "AMD"
    if metal_layers:
      for i in metal_layers: placement[i] = "METAL"
    if cpu_layers:
      for i in cpu_layers: placement[i] = "CPU"
    return placement

  # amd_budget_gb=0 means all-METAL (useful when AMD kernel compilation is unstable)
  if amd_budget_gb <= 0.0:
    print(f"Placement: 0 layers AMD, {num_blocks} layers METAL (AMD disabled via budget=0)")
    return ["METAL"] * num_blocks

  # Auto: full-attn → AMD first, then greedily fill remaining AMD budget with linear-attn
  fai = config.full_attention_interval
  is_full_attn = [(i + 1) % fai == 0 for i in range(num_blocks)] if fai > 0 else [True] * num_blocks
  placement = ["AMD" if fa else "METAL" for fa in is_full_attn]

  amd_used = sum(block_sizes[i] for i in range(num_blocks) if placement[i] == "AMD")
  budget = amd_budget_gb * 1e9

  for i in range(num_blocks):
    if placement[i] == "METAL" and amd_used + block_sizes[i] <= budget:
      placement[i] = "AMD"
      amd_used += block_sizes[i]

  print(f"Placement: {placement.count('AMD')} layers AMD, {placement.count('METAL')} layers METAL "
        f"({amd_used / 1e9:.2f} GB on AMD)")
  return placement


def _patch_for_multidevice(model: Transformer, placement: list[str]) -> None:
  """Patch model.forward and full-attn _init_state for multi-device inference.

  tinygrad requires all buffers in a single kernel to be on the same device.
  This:
  1. Adds explicit .to(device) copies at block device boundaries in forward.
  2. Patches TransformerBlock._init_state so freqs_cis (normally created on
     the default METAL device by precompute_freqs_cis) is moved to the block's
     device before it's captured as an implicit buffer.
  """
  import types
  from tinygrad.llm.model import TransformerBlock

  # Patch full-attn blocks so freqs_cis ends up on the block's device
  for i, block in enumerate(model.blk):
    if not isinstance(block, TransformerBlock):
      continue
    target_dev = placement[i]
    if target_dev == "METAL":
      continue  # freqs_cis is already created on METAL, no patch needed

    orig_init = type(block)._init_state

    def _make_patched_init(dev: str, orig):
      def patched_init(self, x):
        orig(self, x)
        # precompute_freqs_cis creates on the default device (METAL).
        # Move to this block's device so it's on the same device as x.
        if hasattr(self, "freqs_cis") and isinstance(self.freqs_cis.device, str) and self.freqs_cis.device != dev:
          self.freqs_cis = self.freqs_cis.to(dev).contiguous().realize()
      return patched_init

    block._init_state = types.MethodType(_make_patched_init(target_dev, orig_init), block)

  # Get the device of the output projection (non-block tensors land on METAL)
  output_device: str = model.output_norm.weight.device  # type: ignore[assignment]

  def _patched_forward(tokens: Tensor, start_pos, temperature: Tensor) -> Tensor:
    x = model.token_embd(tokens).float()
    for i, block in enumerate(model.blk):
      target = placement[i]
      if x.device != target:
        x = x.to(target)
      x = block(x, start_pos)
    if x.device != output_device:
      x = x.to(output_device)
    logits = model.output(model.output_norm(x))[:, -1, :]
    return (logits / temperature.maximum(1e-12)
            - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  model.forward = _patched_forward  # type: ignore[method-assign]
  # Rebuild JITs so they capture the patched forward
  model.prefill_jit = TinyJit(model.forward)
  model.rollout_jit = TinyJit(model.forward)


def from_safetensors(path, max_context: int = 4096, amd_layers=None, metal_layers=None,
                     cpu_layers=None, amd_budget_gb: float = 20.0) -> Transformer:
  path = pathlib.Path(path)
  header, meta = _read_header(path)

  cfg_dict = json.loads(meta["tinygrad_config"])
  bits = int(meta.get("quant_bits", "0"))

  ssm_dict = cfg_dict.pop("ssm", None)
  ssm = SSMConfig(**ssm_dict) if ssm_dict else None
  config = TransformerConfig(ssm=ssm, max_context=max_context, **cfg_dict)

  model = Transformer(config)
  if bits in (4, 8):
    replace_linears_with_quantized(model, bits=bits, skip={"output"})

  # Per-block byte sizes from the safetensors header (for placement budget)
  block_sizes: dict[int, int] = {}
  for i in range(config.num_blocks):
    prefix = f"blk.{i}."
    block_sizes[i] = sum(
      header[k]["data_offsets"][1] - header[k]["data_offsets"][0]
      for k in header if k.startswith(prefix)
    )

  placement = _resolve_placement(config, amd_layers, metal_layers, cpu_layers, amd_budget_gb, block_sizes)

  raw = nn.state.safe_load(str(path))
  state = nn.state.get_state_dict(model)

  missing = set(state.keys()) - set(raw.keys())
  extra = set(raw.keys()) - set(state.keys())
  if missing: print(f"WARNING: keys missing from safetensors: {sorted(missing)[:5]}")
  if extra:   print(f"NOTE: extra keys in safetensors (skipped): {sorted(extra)[:5]}")

  for k in tqdm(sorted(state.keys()), desc="loading weights"):
    if k not in raw:
      continue
    device = _device_for_key(k, placement)
    state[k].replace(raw[k].to(device))

  Tensor.realize(*list(state.values()))

  # Patch forward for multi-device if blocks span more than one device
  devices_used = set(placement) | {"METAL"}  # METAL for token_embd/output
  if len(devices_used) > 1:
    _patch_for_multidevice(model, placement)

  return model

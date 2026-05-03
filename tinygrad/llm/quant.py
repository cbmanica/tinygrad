from __future__ import annotations
from tinygrad import Tensor, nn
from tinygrad.dtype import dtypes


class QuantizedLinear:
  def __init__(self, in_features: int, out_features: int, bits: int, bias: bool = False):
    assert bits in (4, 8)
    self.bits = bits
    self.in_features = in_features
    self.out_features = out_features
    if bits == 8:
      self.weight = Tensor.zeros(out_features, in_features, dtype=dtypes.int8)
    else:
      self.weight = Tensor.zeros(out_features, (in_features + 1) // 2, dtype=dtypes.uint8)
    self.scale = Tensor.zeros(out_features, dtype=dtypes.float16)
    self.bias: Tensor | None = Tensor.zeros(out_features, dtype=dtypes.float16) if bias else None

  def _dequant(self) -> Tensor:
    if self.bits == 8:
      w = self.weight.cast(dtypes.float16)
    else:
      lo = (self.weight & 0xF).cast(dtypes.int8) - 8
      hi = ((self.weight >> 4) & 0xF).cast(dtypes.int8) - 8
      w = Tensor.stack(lo, hi, dim=-1).reshape(self.out_features, -1)[:, :self.in_features].cast(dtypes.float16)
    return w * self.scale.unsqueeze(1)

  def __call__(self, x: Tensor) -> Tensor:
    return x.linear(self._dequant().T, self.bias)


def replace_linears_with_quantized(module, bits: int, skip: set[str] | None = None, _prefix: str = ""):
  """Recursively replace nn.Linear with QuantizedLinear, skipping names in `skip`."""
  skip = skip or set()
  if not hasattr(module, '__dict__'):
    return
  for attr_name in list(vars(module)):
    attr = getattr(module, attr_name)
    full_path = f"{_prefix}.{attr_name}".lstrip(".")
    if isinstance(attr, nn.Linear):
      if attr_name not in skip:
        ql = QuantizedLinear(attr.weight.shape[1], attr.weight.shape[0], bits=bits,
                             bias=attr.bias is not None)
        setattr(module, attr_name, ql)
    elif isinstance(attr, list):
      for i, item in enumerate(attr):
        if hasattr(item, '__dict__'):
          replace_linears_with_quantized(item, bits, skip, f"{full_path}.{i}")
    elif hasattr(attr, '__dict__') and not isinstance(attr, type) and not isinstance(attr, Tensor):
      replace_linears_with_quantized(attr, bits, skip, full_path)

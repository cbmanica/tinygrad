from typing import NamedTuple
from tinygrad import Tensor, dtypes, nn
from extra.models.unet import timestep_embedding

configs = {
  "flux-schnell": {
    "in_channels": 64, "vec_in_dim": 768, "context_in_dim": 4096,
    "hidden_size": 3072, "mlp_ratio": 4.0, "num_heads": 24,
    "depth": 19, "depth_single_blocks": 38,
    "axes_dim": [16, 56, 56], "theta": 10000,
    "qkv_bias": True, "guidance_embed": False,
  },
  "flux-dev": {
    "in_channels": 64, "vec_in_dim": 768, "context_in_dim": 4096,
    "hidden_size": 3072, "mlp_ratio": 4.0, "num_heads": 24,
    "depth": 19, "depth_single_blocks": 38,
    "axes_dim": [16, 56, 56], "theta": 10000,
    "qkv_bias": True, "guidance_embed": True,
  },
}

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
  assert dim % 2 == 0
  half = dim // 2
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:half].cast(dtypes.float32) / dim))
  # pos: (B, N)  freqs: (half,)
  angles = pos.cast(dtypes.float32).unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (B, N, half)
  cos = angles.cos()
  sin = angles.sin()
  # build 2x2 rotation matrices: [[cos, -sin], [sin, cos]]
  return Tensor.stack(cos, -sin, sin, cos, dim=-1).reshape(*pos.shape, half, 2, 2)

def EmbedND(pos: Tensor, axes_dim: list, theta: int) -> Tensor:
  # pos: (B, N, num_axes)
  embs = [rope(pos[:, :, i], axes_dim[i], theta) for i in range(len(axes_dim))]
  return Tensor.cat(*embs, dim=-3)  # (B, N, head_dim//2, 2, 2)

def apply_rope(q: Tensor, k: Tensor, pe: Tensor) -> tuple:
  # q, k: (B, n_heads, N, head_dim)
  # pe:   (B, N, head_dim//2, 2, 2)
  def rot(x):
    B, H, N, D = x.shape
    x = x.reshape(B, H, N, D // 2, 2, 1)
    r = pe.unsqueeze(1)  # (B, 1, N, D//2, 2, 2)
    return r.matmul(x).squeeze(-1).flatten(-2)  # (B, H, N, D)
  return rot(q), rot(k)


class MLPEmbedder:
  def __init__(self, in_dim: int, hidden_dim: int):
    self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
    self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
  def __call__(self, x: Tensor) -> Tensor:
    return self.out_layer(self.in_layer(x).silu())


class FluxRMSNorm:
  def __init__(self, dim: int, eps: float = 1e-6):
    self.scale = Tensor.ones(dim)
    self.eps = eps
  def __call__(self, x: Tensor) -> Tensor:
    return x * Tensor.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).cast(x.dtype) * self.scale


class QKNorm:
  def __init__(self, dim: int):
    self.query_norm = FluxRMSNorm(dim)
    self.key_norm = FluxRMSNorm(dim)
  def __call__(self, q: Tensor, k: Tensor) -> tuple:
    return self.query_norm(q), self.key_norm(k)


class SelfAttention:
  def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.norm = QKNorm(self.head_dim)
    self.proj = nn.Linear(dim, dim)

  def pre_attn(self, x: Tensor) -> tuple:
    B, N, _ = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, N, H, D)
    q, k = self.norm(q, k)
    q = q.transpose(1, 2)  # (B, H, N, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    return q, k, v

  def __call__(self, x: Tensor, pe: Tensor) -> Tensor:
    B, N, dim = x.shape
    q, k, v = self.pre_attn(x)
    q, k = apply_rope(q, k, pe)
    attn = q.scaled_dot_product_attention(k, v)
    return self.proj(attn.transpose(1, 2).reshape(B, N, dim))


class ModulationOut(NamedTuple):
  shift: Tensor
  scale: Tensor
  gate: Tensor


class Modulation:
  def __init__(self, dim: int, double: bool):
    self.is_double = double
    self.multiplier = 6 if double else 3
    self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

  def __call__(self, vec: Tensor) -> tuple:
    out = self.lin(vec.silu()).chunk(self.multiplier, dim=-1)
    m0 = ModulationOut(out[0], out[1], out[2])
    m1 = ModulationOut(out[3], out[4], out[5]) if self.is_double else None
    return m0, m1


def _modulate(x: Tensor, m: ModulationOut) -> Tensor:
  return x * (1 + m.scale.unsqueeze(1)) + m.shift.unsqueeze(1)


class DoubleStreamBlock:
  def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
    mlp_hidden = int(hidden_size * mlp_ratio)
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.img_mod = Modulation(hidden_size, double=True)
    self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
    self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_mlp = [nn.Linear(hidden_size, mlp_hidden, bias=True), Tensor.gelu, nn.Linear(mlp_hidden, hidden_size, bias=True)]
    self.txt_mod = Modulation(hidden_size, double=True)
    self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
    self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_mlp = [nn.Linear(hidden_size, mlp_hidden, bias=True), Tensor.gelu, nn.Linear(mlp_hidden, hidden_size, bias=True)]

  def __call__(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple:
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    # pre-norm + modulate
    img_normed = _modulate(self.img_norm1(img), img_mod1)
    txt_normed = _modulate(self.txt_norm1(txt), txt_mod1)

    # compute Q, K, V for each stream
    img_q, img_k, img_v = self.img_attn.pre_attn(img_normed)
    txt_q, txt_k, txt_v = self.txt_attn.pre_attn(txt_normed)

    # apply rope to both sets using shared pe
    img_q, img_k = apply_rope(img_q, img_k, pe[:, txt.shape[1]:])
    txt_q, txt_k = apply_rope(txt_q, txt_k, pe[:, :txt.shape[1]])

    # joint attention: concatenate along sequence dim
    q = Tensor.cat(txt_q, img_q, dim=2)
    k = Tensor.cat(txt_k, img_k, dim=2)
    v = Tensor.cat(txt_v, img_v, dim=2)
    attn = q.scaled_dot_product_attention(k, v)  # (B, H, N_txt+N_img, D)

    # split and project separately
    N_txt = txt.shape[1]
    txt_attn_out = attn[:, :, :N_txt].transpose(1, 2).reshape(txt.shape[0], N_txt, self.hidden_size)
    img_attn_out = attn[:, :, N_txt:].transpose(1, 2).reshape(img.shape[0], img.shape[1], self.hidden_size)

    img = img + img_mod1.gate.unsqueeze(1) * self.img_attn.proj(img_attn_out)
    txt = txt + txt_mod1.gate.unsqueeze(1) * self.txt_attn.proj(txt_attn_out)

    # FFN for each stream
    img = img + img_mod2.gate.unsqueeze(1) * _modulate(self.img_norm2(img), img_mod2).sequential(self.img_mlp)
    txt = txt + txt_mod2.gate.unsqueeze(1) * _modulate(self.txt_norm2(txt), txt_mod2).sequential(self.txt_mlp)

    return img, txt


class SingleStreamBlock:
  def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    mlp_hidden = int(hidden_size * mlp_ratio)
    self.mlp_hidden = mlp_hidden
    self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden, bias=True)
    self.linear2 = nn.Linear(hidden_size + mlp_hidden, hidden_size, bias=True)
    self.norm = QKNorm(self.head_dim)
    self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.modulation = Modulation(hidden_size, double=False)

  def __call__(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
    mod, _ = self.modulation(vec)
    x_normed = _modulate(self.pre_norm(x), mod)

    # combined QKV + MLP gate projection
    qkv, mlp_gate = self.linear1(x_normed).split([self.hidden_size * 3, self.mlp_hidden], dim=-1)

    B, N, _ = x.shape
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
    q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
    q, k = self.norm(q, k)
    q, k = apply_rope(q, k, pe)

    attn = q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(B, N, self.hidden_size)
    out = Tensor.cat(attn, mlp_gate.gelu(), dim=-1)
    x = x + mod.gate.unsqueeze(1) * self.linear2(out)
    return x


class LastLayer:
  def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    self.adaLN_modulation = [Tensor.silu, nn.Linear(hidden_size, 2 * hidden_size, bias=True)]

  def __call__(self, x: Tensor, vec: Tensor) -> Tensor:
    shift, scale = vec.sequential(self.adaLN_modulation).chunk(2, dim=-1)
    x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return self.linear(x)


class Flux:
  def __init__(self, params: dict):
    self.params = params
    self.in_channels = params["in_channels"]
    self.out_channels = self.in_channels
    self.hidden_size = params["hidden_size"]
    self.num_heads = params["num_heads"]
    self.axes_dim = params["axes_dim"]
    self.theta = params["theta"]

    self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
    self.time_in = MLPEmbedder(256, self.hidden_size)
    self.vector_in = MLPEmbedder(params["vec_in_dim"], self.hidden_size)
    self.guidance_in = MLPEmbedder(256, self.hidden_size) if params["guidance_embed"] else None
    self.txt_in = nn.Linear(params["context_in_dim"], self.hidden_size, bias=True)

    self.double_blocks = [
      DoubleStreamBlock(self.hidden_size, self.num_heads, params["mlp_ratio"], params["qkv_bias"])
      for _ in range(params["depth"])
    ]
    self.single_blocks = [
      SingleStreamBlock(self.hidden_size, self.num_heads, params["mlp_ratio"])
      for _ in range(params["depth_single_blocks"])
    ]
    self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

  def __call__(self, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor,
               timesteps: Tensor, y: Tensor, guidance: Tensor | None = None) -> Tensor:
    img = self.img_in(img)
    txt = self.txt_in(txt)

    vec = self.time_in(timestep_embedding(timesteps, 256).cast(img.dtype))
    if self.guidance_in is not None:
      assert guidance is not None, "flux-dev requires guidance tensor"
      vec = vec + self.guidance_in(timestep_embedding(guidance, 256).cast(img.dtype))
    vec = vec + self.vector_in(y)

    # build position embeddings for full sequence (txt first, then img)
    ids = Tensor.cat(txt_ids, img_ids, dim=1)
    pe = EmbedND(ids, self.axes_dim, self.theta)

    for block in self.double_blocks:
      img, txt = block(img, txt, vec, pe)

    x = Tensor.cat(txt, img, dim=1)
    for block in self.single_blocks:
      x = block(x, vec, pe)

    img = x[:, txt.shape[1]:]
    return self.final_layer(img, vec)

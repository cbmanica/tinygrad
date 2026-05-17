import argparse, math, os, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from tinygrad import Tensor, dtypes, nn, GlobalCounters
from tinygrad.helpers import DEV
from tinygrad.helpers import fetch, Timing, trange, getenv
from tinygrad.engine.jit import TinyJit

def hf_fetch(url: str, name: str) -> Path:
  token = os.environ.get("HF_TOKEN", "")
  needs_auth = "black-forest-labs" in url
  headers = {"Authorization": f"Bearer {token}"} if (token and needs_auth) else {}
  return fetch(url, name, headers=headers)
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from extra.models.flux import Flux, configs
from extra.models.clip import Closed, Tokenizer
from extra.models.t5 import T5EncoderModel, T5Tokenizer, T5Config
from extra.models.unet import timestep_embedding
from examples.sdxl import FirstStageModel

# ---------------------------------------------------------------------------
# Weight URLs (all from FLUX.1-schnell for non-gated access; ae works for dev too)
# ---------------------------------------------------------------------------
FLUX_SCHNELL_URL = "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors"
FLUX_DEV_URL     = "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"
VAE_URL          = "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors"
CLIP_URL         = "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder/model.safetensors"
T5_SHARD1_URL    = "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder_2/model-00001-of-00002.safetensors"
T5_SHARD2_URL    = "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder_2/model-00002-of-00002.safetensors"
T5_TOKENIZER_URL = "https://huggingface.co/google-t5/t5-large/resolve/main/spiece.model"


# ---------------------------------------------------------------------------
# CLIP-L with text_projection — matches HuggingFace CLIPTextModelWithProjection
# key structure: text_model.* and text_projection.weight
# ---------------------------------------------------------------------------
class CLIPTextEncoderL:
  def __init__(self):
    self.text_model = Closed.ClipTextTransformer()

  def encode(self, tokens: Tensor) -> Tensor:
    x = self.text_model(tokens)                              # (B, 77, 768)
    eos_pos = (tokens == 49407).int().argmax(axis=-1)        # position of EOT token
    return x[Tensor.arange(tokens.shape[0]), eos_pos]       # (B, 768) pooler_output, no projection


# ---------------------------------------------------------------------------
# Flux VAE — LDM-style with 16-channel latents
# ---------------------------------------------------------------------------
class FluxVAE:
  SCALING_FACTOR = 0.3611
  SHIFT_FACTOR   = 0.1159

  def __init__(self):
    self.model = FirstStageModel(
      embed_dim=16, ch=128, in_ch=3, out_ch=3, z_ch=16,
      ch_mult=[1, 2, 4, 4], num_res_blocks=2, resolution=256,
    )

  def decode(self, z: Tensor) -> Tensor:
    z = z / self.SCALING_FACTOR + self.SHIFT_FACTOR
    return self.model.decoder(z)  # Flux AE has no post_quant_conv


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
def patchify(x: Tensor, patch_size: int = 2):
  B, C, H, W = x.shape
  pH, pW = H // patch_size, W // patch_size
  # (B, C, pH, p, pW, p) → (B, pH, pW, C, p, p) → (B, pH*pW, C*p*p)
  x = x.reshape(B, C, pH, patch_size, pW, patch_size)
  x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, pH * pW, C * patch_size * patch_size)
  # position ids: [time=0, h_idx, w_idx]
  h_idx = Tensor.arange(pH, dtype=dtypes.int32).reshape(1, pH, 1).expand(B, pH, pW).reshape(B, pH * pW)
  w_idx = Tensor.arange(pW, dtype=dtypes.int32).reshape(1, 1, pW).expand(B, pH, pW).reshape(B, pH * pW)
  t_idx = Tensor.zeros(B, pH * pW, dtype=dtypes.int32)
  img_ids = Tensor.stack(t_idx, h_idx, w_idx, dim=-1)
  return x, img_ids


def unpatchify(x: Tensor, h: int, w: int, patch_size: int = 2, out_channels: int = 16) -> Tensor:
  B = x.shape[0]
  pH, pW = h // patch_size, w // patch_size
  x = x.reshape(B, pH, pW, out_channels, patch_size, patch_size)
  return x.permute(0, 3, 1, 4, 2, 5).reshape(B, out_channels, h, w)


def make_txt_ids(seq_len: int, batch: int) -> Tensor:
  return Tensor.zeros(batch, seq_len, 3, dtype=dtypes.int32)


# ---------------------------------------------------------------------------
# Text encoding — runs T5 then CLIP sequentially to minimise peak VRAM
# ---------------------------------------------------------------------------
def encode_prompt(prompt: str, model_dtype, debug: bool = False) -> tuple:
  # --- T5-XXL ---
  print("loading T5 tokenizer...")
  spiece = hf_fetch(T5_TOKENIZER_URL, "flux_t5_spiece.model")
  tokenizer = T5Tokenizer(spiece)
  tokens = Tensor(tokenizer.encode(prompt, 256)).reshape(1, -1)  # (1, 256)

  print("loading T5 weights (2 shards)...")
  t5 = T5EncoderModel(T5Config())
  shard1 = safe_load(hf_fetch(T5_SHARD1_URL, "flux_t5_shard1.safetensors"))
  shard2 = safe_load(hf_fetch(T5_SHARD2_URL, "flux_t5_shard2.safetensors"))
  t5_weights = {**shard1, **shard2}
  load_state_dict(t5, t5_weights, strict=False, verbose=False, realize=False)
  Tensor.realize(*nn.state.get_parameters(t5))
  del t5_weights, shard1, shard2

  def print_tensor_stats(name, t):
    a = t.float().numpy()
    print(f"  {name}: {t.shape}, norm={float(np.linalg.norm(a)):.2f}, mean={a.mean():.4f}, std={a.std():.4f}")

  print("encoding with T5...")
  t5_emb = t5(tokens).cast(model_dtype).realize()  # (1, 256, 4096)
  del t5
  if debug: print_tensor_stats("t5_emb", t5_emb)

  # --- CLIP-L ---
  print("loading CLIP weights...")
  clip = CLIPTextEncoderL()
  clip_weights = safe_load(hf_fetch(CLIP_URL, "flux_clip_l.safetensors"))
  load_state_dict(clip, clip_weights, strict=False, verbose=False, realize=False)
  Tensor.realize(*nn.state.get_parameters(clip))
  del clip_weights

  print("encoding with CLIP...")
  clip_tokenizer = Tokenizer.ClipTokenizer()
  clip_tokens = Tensor(clip_tokenizer.encode(prompt)).reshape(1, -1)  # (1, 77)
  clip_pooled = clip.encode(clip_tokens).cast(model_dtype).realize()  # (1, 768)
  del clip
  if debug: print_tensor_stats("clip_pooled", clip_pooled)

  return t5_emb, clip_pooled


# ---------------------------------------------------------------------------
# Euler sampler (rectified flow)
# ---------------------------------------------------------------------------
def euler_sample(model: Flux, latents: Tensor, img_ids: Tensor,
                 t5_emb: Tensor, txt_ids: Tensor, clip_pooled: Tensor,
                 num_steps: int, guidance_scale: float, model_name: str, timing: bool, debug: bool = False) -> Tensor:
  timesteps = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float32)

  @TinyJit
  def jit_step(latents, t_vec):
    return model(latents, img_ids, t5_emb, txt_ids, t_vec, clip_pooled)

  @TinyJit
  def jit_step_dev(latents, t_vec, g_vec):
    return model(latents, img_ids, t5_emb, txt_ids, t_vec, clip_pooled, g_vec)

  for i in trange(num_steps):
    with Timing("  step: ", enabled=timing, on_exit=lambda _: f", VRAM {GlobalCounters.mem_used/1e9:.2f} GB"):
      t_curr = float(timesteps[i])
      t_next = float(timesteps[i + 1])
      t_vec  = Tensor.full((latents.shape[0],), t_curr * 1000.0, dtype=latents.dtype).contiguous()

      if model_name == "flux-dev":
        g_vec = Tensor.full((latents.shape[0],), guidance_scale, dtype=latents.dtype).contiguous()
        velocity = jit_step_dev(latents, t_vec, g_vec)
      else:
        velocity = jit_step(latents, t_vec)

      if debug:
        v_np = velocity.float().numpy()
        l_np = latents.float().numpy()
        spatial_std = v_np[0].mean(axis=-1).std()
        # per-channel velocity means: v_np[0] is (N, 64), 64 = 16ch × 4 sub-pixels
        v_ch = v_np[0].reshape(-1, 16, 4)         # (N, 16, 4)
        ch_v_means = v_ch.mean(axis=(0, 2))        # (16,) mean per channel
        ch_v_spatial = v_ch.mean(axis=2).std(axis=0)  # (16,) within-channel spatial std
        print(f"    t={t_curr:.2f}→{t_next:.2f}  vel std={v_np.std():.3f} mean={v_np.mean():.3f}  lat std={l_np.std():.3f}  vel spatial_std={spatial_std:.4f}")
        print(f"    ch vel means: {' '.join(f'{m:+.2f}' for m in ch_v_means)}")
        print(f"    ch vel spstd: {' '.join(f'{s:.3f}' for s in ch_v_spatial)}")
      latents = (latents + (t_next - t_curr) * velocity).realize()

  if debug:
    l = latents.numpy()
    print(f"  latent stats: min={l.min():.3f} max={l.max():.3f} mean={l.mean():.3f} std={l.std():.3f}")
  return latents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Flux.1 image generation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--model",    type=str,   default="flux-schnell", choices=["flux-schnell", "flux-dev"])
  parser.add_argument("--prompt",   type=str,   required=True)
  parser.add_argument("--steps",    type=int,   default=None,  help="diffusion steps (default: 4 for schnell, 20 for dev)")
  parser.add_argument("--guidance", type=float, default=3.5,   help="guidance scale (flux-dev only)")
  parser.add_argument("--width",    type=int,   default=1024,  help="output width (multiple of 16)")
  parser.add_argument("--height",   type=int,   default=1024,  help="output height (multiple of 16)")
  parser.add_argument("--seed",     type=int,   default=None)
  parser.add_argument("--dtype",    type=str,   default="bfloat16", choices=["float32", "bfloat16", "float16"],
                                                help="model dtype (float32 if bfloat16 has distortion on AMD)")
  parser.add_argument("--out",      type=str,   default=str(Path(tempfile.gettempdir()) / "flux_out.png"))
  parser.add_argument("--timing",   action="store_true")
  parser.add_argument("--noshow",   action="store_true")
  parser.add_argument("--debug",    action="store_true", help="print embedding/latent/velocity diagnostics")
  parser.add_argument("--device",   type=str,   default=None, help="tinygrad device override (e.g. AMD, METAL, GPU)")
  args = parser.parse_args()

  if args.device is not None:
    DEV.value = args.device

  assert args.width  % 16 == 0, f"width must be multiple of 16, got {args.width}"
  assert args.height % 16 == 0, f"height must be multiple of 16, got {args.height}"

  if args.seed is not None:
    Tensor.manual_seed(args.seed)

  num_steps  = args.steps or (4 if args.model == "flux-schnell" else 20)
  model_dtype = getattr(dtypes, args.dtype)

  print(f"generating {args.width}×{args.height} with {args.model}, {num_steps} steps, dtype={args.dtype}")
  print(f"prompt: {args.prompt}")

  # Step 1: encode text (sequential to save VRAM)
  t5_emb, clip_pooled = encode_prompt(args.prompt, model_dtype, debug=args.debug)

  # Step 2: load Flux transformer and sample
  print(f"\nloading Flux transformer ({args.model})...")
  flux = Flux(configs[args.model])
  flux_url = FLUX_DEV_URL if args.model == "flux-dev" else FLUX_SCHNELL_URL
  flux_name = f"flux1-{args.model.split('-')[1]}.safetensors"
  flux_weights = safe_load(hf_fetch(flux_url, flux_name))

  if args.debug:
    model_keys = set(nn.state.get_state_dict(flux).keys())
    weight_keys = set(flux_weights.keys())
    matched = model_keys & weight_keys
    missing = model_keys - weight_keys
    extra   = weight_keys - model_keys
    print(f"  key match: {len(matched)}/{len(model_keys)} model keys found in weights file")
    if missing: print(f"  MISSING ({len(missing)}): {sorted(missing)[:5]}{'...' if len(missing)>5 else ''}")
    if extra:   print(f"  EXTRA   ({len(extra)}): {sorted(extra)[:5]}{'...' if len(extra)>5 else ''}")

  with Timing("loaded transformer in "):
    load_state_dict(flux, flux_weights, strict=False, verbose=False, realize=False)
    Tensor.realize(*nn.state.get_parameters(flux))
  del flux_weights
  print(f"  VRAM after load: {GlobalCounters.mem_used/1e9:.2f} GB")

  latent_h, latent_w = args.height // 8, args.width // 8  # VAE downsample factor = 8
  latents = Tensor.randn(1, 16, latent_h, latent_w, dtype=model_dtype)
  latents, img_ids = patchify(latents)          # (1, N, 64), (1, N, 3)
  txt_ids = make_txt_ids(256, 1)
  if args.debug: print(f"  img_ids sample (first 5 patches): {img_ids[0, :5].numpy().tolist()}")

  print(f"\nsampling ({num_steps} steps)...")
  with Timing("sampling done in "):
    latents = euler_sample(flux, latents, img_ids, t5_emb, txt_ids, clip_pooled,
                           num_steps, args.guidance, args.model, args.timing, debug=args.debug)
  del flux

  # Step 3: decode latents with VAE
  print("\nloading VAE and decoding...")
  latents = unpatchify(latents, latent_h, latent_w)  # (1, 16, H//8, W//8)

  vae = FluxVAE()
  vae_weights = safe_load(hf_fetch(VAE_URL, "flux_ae.safetensors"))
  load_state_dict(vae.model, vae_weights, strict=False, verbose=False, realize=False)
  Tensor.realize(*nn.state.get_parameters(vae.model))
  del vae_weights

  if args.debug:
    lat_np = latents.numpy()
    n_ch = lat_np.shape[1]
    ch_means = np.array([lat_np[0, ch].mean() for ch in range(n_ch)])
    ch_stds  = np.array([lat_np[0, ch].std()  for ch in range(n_ch)])
    print("  per-channel latent means/stds:")
    for ch in range(n_ch):
      print(f"    ch{ch:2d}: mean={ch_means[ch]:.3f}  std={ch_stds[ch]:.3f}")

    # Fake latent: same per-channel mean/std as diffusion output, but spatially random.
    # If this also decodes all-dark → channel biases alone cause it (not spatial structure).
    fake_latent = (np.random.randn(*lat_np.shape).astype(np.float32)
                   * ch_stds[None, :, None, None] + ch_means[None, :, None, None])
    fake_out = vae.decode(Tensor(fake_latent)).realize().numpy()
    print(f"  fake-latent decode: min={fake_out.min():.3f} max={fake_out.max():.3f} mean={fake_out.mean():.3f}")

    # Zero-mean latent: same spatial std but channels centered at 0.
    # If this decodes to a mid-gray image → per-channel means ARE the bug.
    zero_mean_latent = (np.random.randn(*lat_np.shape).astype(np.float32)
                        * ch_stds[None, :, None, None])  # mean=0 per channel
    zero_out = vae.decode(Tensor(zero_mean_latent)).realize().numpy()
    print(f"  zero-mean decode:   min={zero_out.min():.3f} max={zero_out.max():.3f} mean={zero_out.mean():.3f}")

  with Timing("decoded in "):
    image = vae.decode(latents.cast(dtypes.float32)).realize()  # (1, 3, H, W)

  if args.debug:
    img_np = image.numpy()
    print(f"  vae output stats: min={img_np.min():.3f} max={img_np.max():.3f} mean={img_np.mean():.3f} per-channel means: R={img_np[0,0].mean():.3f} G={img_np[0,1].mean():.3f} B={img_np[0,2].mean():.3f}")
  image = ((image.clamp(-1, 1) + 1) / 2 * 255).cast(dtypes.uint8)
  image = image[0].permute(1, 2, 0).numpy()

  im = Image.fromarray(image)
  im.save(args.out)
  print(f"\nsaved → {args.out}")
  if not args.noshow:
    im.show()

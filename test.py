from tinygrad import Tensor, Device, Context

# The new 2026 way to set the device for a block of code
with Context(DEV="AMD"):
    t = Tensor([1, 2, 3]).realize()
    print(f"Device: {t.device}")
    print(f"Default Device: {Device.DEFAULT}")

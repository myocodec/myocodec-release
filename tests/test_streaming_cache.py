"""Verify the bounded cache: identical outputs, bounded memory, still no lookahead."""
import sys, torch
from streaming_emg_codec.model.attention import CausalSelfAttention, InferenceCache

# Runs on GPU when available and on CPU otherwise; CPU has no bfloat16 SDPA kernel worth
# using here, so the cache dtype follows the device.
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEV.type == "cuda" else torch.float32
TOL = 2e-2 if DTYPE is torch.bfloat16 else 1e-4

torch.manual_seed(0)
H, D, W = 4, 32, 128
B, T = 8, 900
attn = CausalSelfAttention(n_embd=H * D, n_heads=H, head_dim=D, dropout_rate=0.0,
                           window_size=W, rotary_base=10000.0).eval().to(DEV)
x = torch.randn(B, T, H * D, device=DEV)
ok = True

def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

# ---- 1. streamed == full forward (the property that must not change) ----
print("\n1. streaming equivalence (T=900 frames, well past the 128-frame window)")
with torch.no_grad():
    full = attn(x)
    for step in (1, 25, 128, 300):
        cache = InferenceCache(batch_size=B, max_seqlen=W, n_heads=H, head_dim=D,
                               device=DEV, dtype=DTYPE)
        out = torch.cat([attn.decode(x[:, i:i + step], cache) for i in range(0, T, step)], dim=1)
        d = (out - full).abs().max().item()
        check(f"chunk={step:3d}: streamed vs full forward", d < TOL, f"max|d|={d:.2e}")

# ---- 2. memory is now bounded ----
print("\n2. memory bound (was O(stream length): 4096 frames after 80 s)")
cache = InferenceCache(batch_size=32, max_seqlen=W, n_heads=H, head_dim=D,
                       device=DEV, dtype=DTYPE)
sizes = []
with torch.no_grad():
    for i in range(0, 8000, 25):
        attn.decode(torch.randn(32, 25, H * D, device=DEV), cache)
        sizes.append(cache.max_seqlen)
mb = cache.kv_cache.numel() * cache.kv_cache.element_size() / 1e6
check(f"buffer stayed bounded over 8000 frames (160 s)", max(sizes) <= 2 * W + 25,
      f"max_seqlen={max(sizes)} (cap {2*W+25}), {mb:.2f} MB, absolute pos={cache.seqlen_offset}")
check("absolute position still tracked for RoPE", cache.seqlen_offset == 8000,
      f"seqlen_offset={cache.seqlen_offset}")

# ---- 3. still strictly causal: no lookahead ----
print("\n3. no lookahead (perturb frame t, earlier outputs must be identical)")
t = 400
x2 = x.clone(); x2[:, t] += torch.randn(B, H * D, device=DEV) * 5.0
with torch.no_grad():
    c1 = InferenceCache(batch_size=B, max_seqlen=W, n_heads=H, head_dim=D,
                        device=DEV, dtype=DTYPE)
    c2 = InferenceCache(batch_size=B, max_seqlen=W, n_heads=H, head_dim=D,
                        device=DEV, dtype=DTYPE)
    o1 = torch.cat([attn.decode(x[:, i:i + 25], c1) for i in range(0, T, 25)], dim=1)
    o2 = torch.cat([attn.decode(x2[:, i:i + 25], c2) for i in range(0, T, 25)], dim=1)
before = (o1[:, :t] - o2[:, :t]).abs().max().item()
after = (o1[:, t:] - o2[:, t:]).abs().max().item()
check("frames < t bit-identical (no lookahead)", before == 0.0, f"max|d|={before:.2e}")
check("frames >= t do change", after > 1e-3, f"max|d|={after:.2e}")

# ---- 4. window is genuinely bounded: frames older than W have no influence ----
print("\n4. window really is 128 (a frame >W back must not affect the current output)")
with torch.no_grad():
    c1 = InferenceCache(batch_size=B, max_seqlen=W, n_heads=H, head_dim=D,
                        device=DEV, dtype=DTYPE)
    c2 = InferenceCache(batch_size=B, max_seqlen=W, n_heads=H, head_dim=D,
                        device=DEV, dtype=DTYPE)
    o1 = torch.cat([attn.decode(x[:, i:i + 1], c1) for i in range(T)], dim=1)
    o2 = torch.cat([attn.decode(x2[:, i:i + 1], c2) for i in range(T)], dim=1)
far = (o1[:, t + W + 2:] - o2[:, t + W + 2:]).abs().max().item()
check(f"outputs beyond t+{W} unaffected (single-layer window holds)", far == 0.0,
      f"max|d|={far:.2e}")

print("\n" + ("ALL PASSED" if ok else "FAILURES PRESENT"))
sys.exit(0 if ok else 1)

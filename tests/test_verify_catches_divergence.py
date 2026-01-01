"""Negative control for the bitwise-equivalence check.

`tools/bench_streaming.py --verify` reports that `StreamingSession` reproduces
`model.streaming_step` bit for bit, and that result is what licenses quoting the fast
path's latencies as the codec's. A check that has only ever passed is not evidence: it
could pass because it compares a tensor with itself, because the comparison set is empty,
or because it is insensitive to the kind of error it exists to catch.

So break the fast path on purpose, several ways, and require the check to fail each time.
A mutation that survives is a hole in the check, not a harmless difference.

    python tests/test_verify_catches_divergence.py

Needs CUDA. Correctness only -- it does not time anything, so a busy GPU is fine.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming_emg_codec.config import load_config                       # noqa: E402
from streaming_emg_codec.model import StreamingEMGCodec                  # noqa: E402
from streaming_emg_codec.model.codec import StreamingState              # noqa: E402
from streaming_emg_codec.model.fast_stream import StreamingSession      # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else None
FRAMES = 96
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def reference(model, frames):
    st, toks, recs = StreamingState(), [], []
    with torch.no_grad():
        for f in frames:
            r, i, st = model.streaming_step(f, state=st, n_codebooks=None)
            toks.append(i.clone()); recs.append(r.clone())
    return torch.cat(toks, dim=2), torch.cat(recs, dim=-1)


def fast(model, frames, mutate=None):
    sess = StreamingSession(model, batch_size=frames[0].shape[0] * frames[0].shape[1],
                            device=frames[0].device, use_cuda_graph=False)
    toks, recs, applied = [], [], [False]
    with torch.no_grad():
        for n, f in enumerate(frames):
            if mutate and not applied[0] and sess._ready:
                mutate(sess); applied[0] = True      # break it once the fast path is live
            r, i = sess.step(f)
            toks.append(i.clone()); recs.append(r.clone())
    return torch.cat(toks, dim=2), torch.cat(recs, dim=-1), sess.warmup_frames, applied[0]


def compare(ref_t, ref_r, got_t, got_r, warm):
    sl = slice(warm, None)
    n = ref_t[:, :, sl].numel()
    tok_match = (got_t[:, :, sl] == ref_t[:, :, sl]).float().mean().item()
    rel = ((got_r - ref_r).norm() / ref_r.norm()).item()
    return tok_match, rel, n


def main():
    if not torch.cuda.is_available():
        print("  [SKIP] needs CUDA")
        return 0
    dev = torch.device("cuda")
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "pretrain_stage2.yaml")
    model = StreamingEMGCodec(cfg.model).to(dev).eval()
    if CKPT:
        model.load_state_dict(torch.load(CKPT, map_location=dev)["model"])
    torch.manual_seed(0)
    frames = [torch.randn(1, 1, cfg.model.frame_size, device=dev) for _ in range(FRAMES)]

    ref_t, ref_r = reference(model, frames)
    got_t, got_r, warm, _ = fast(model, frames)
    m, rel, n = compare(ref_t, ref_r, got_t, got_r, warm)

    print(f"\n0. the check is comparing something ({FRAMES} frames, warmup {warm})")
    check("steady-state comparison set is non-empty", n > 0, f"{n} tokens compared")
    check("reference and fast outputs are distinct tensors",
          ref_t.data_ptr() != got_t.data_ptr() and ref_r.data_ptr() != got_r.data_ptr())
    check("reference output is not degenerate", ref_r.abs().max().item() > 0,
          f"max|ref| = {ref_r.abs().max().item():.3f}")

    print("\n1. unmutated: the paths agree bitwise")
    check("tokens identical", m == 1.0, f"match={m:.6f}")
    check("reconstruction identical", rel == 0.0, f"rel_l2={rel:.2e}")

    print("\n2. each deliberate break must be caught")

    def m_rope(s):      # off-by-one stream position: a plausible bookkeeping bug
        for t in s._rope_pos: s._rope_pos[t] += 1.0

    def m_seqlen(s):    # off-by-one KV length
        if s.backend == "flash": s._seqlens["enc"].add_(1)
        else: s._slot["enc"].add_(1)

    def m_wipekv(s):    # layer 0 attends to a wiped cache: wrong CONTENT, right shape.
        # Swapping the buffer for a clone is a no-op -- writes and reads both follow it --
        # so the content has to be corrupted, not the identity of the tensor.
        (s._kc["enc"][0] if s.backend == "flash" else s._rings["enc"][0]).zero_()

    def m_weight(s):    # perturb the encoder enough to move tokens, not just the latent
        s.model.rvq_in.weight.data[0, 0] += 1.0

    def m_subthreshold(s):   # NOT expected to be caught: see the note below
        s.model.rvq_in.weight.data[0, 0] += 1e-3

    for name, fn in (("stream position off by one", m_rope),
                     ("KV cache length off by one", m_seqlen),
                     ("layer-0 KV cache content wiped", m_wipekv),
                     ("rvq_in weight perturbed by 1.0", m_weight)):
        snap = {k: v.clone() for k, v in model.state_dict().items()}
        g_t, g_r, w2, applied = fast(model, frames, mutate=fn)
        mm, rr, _ = compare(ref_t, ref_r, g_t, g_r, w2)
        model.load_state_dict(snap)
        caught = (mm != 1.0) or (rr != 0.0)
        check(f"caught: {name}", caught and applied,
              f"tokens={mm:.6f} rel_l2={rr:.2e}" + ("" if applied else "  (MUTATION NEVER APPLIED)"))

    print("\n3. what the check is NOT sensitive to, and why that is correct")
    # The decoder consumes only the quantized codes, so the reconstruction is a pure
    # function of the tokens. A perturbation too small to move any token is therefore
    # invisible in BOTH outputs -- not a blind spot in the check, but the quantizer doing
    # its job. It also means the token comparison is the load-bearing signal and the
    # reconstruction comparison is a consequence of it, not independent evidence.
    snap = {k: v.clone() for k, v in model.state_dict().items()}
    g_t, g_r, w2, applied = fast(model, frames, mutate=m_subthreshold)
    mm, rr, _ = compare(ref_t, ref_r, g_t, g_r, w2)
    model.load_state_dict(snap)
    check("a sub-threshold perturbation changes no token, and so no output",
          applied and mm == 1.0 and rr == 0.0,
          f"tokens={mm:.6f} rel_l2={rr:.2e} -- quantization absorbs it")

    print("\n" + ("ALL PASSED" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

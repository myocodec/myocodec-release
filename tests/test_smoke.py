"""End-to-end smoke test: no data, no GPU required.

Builds the paper's model from the released config, runs a few training steps against
synthetic windows, checkpoints, resumes, and checks the streaming path. Finishes in well
under a minute on CPU. It verifies that the code runs and stays self-consistent -- it says
nothing about reconstruction quality, which needs the real corpus and `tools/eval_recon.py`.

    python tests/test_smoke.py
"""
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from streaming_emg_codec.config import load_config                     # noqa: E402
from streaming_emg_codec.data import build_dataset                     # noqa: E402
from streaming_emg_codec.losses import CodecLoss                       # noqa: E402
from streaming_emg_codec.model import EMGDiscriminator, StreamingEMGCodec  # noqa: E402
from streaming_emg_codec.model.codec import StreamingState             # noqa: E402
from streaming_emg_codec.utils import count_parameters, set_seed       # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def main():
    set_seed(0)
    cfg = load_config(REPO / "configs" / "pretrain_stage1.yaml")

    # shrink to something that runs on a laptop; architecture is otherwise the paper's
    cfg.data.kind = "synthetic"
    cfg.data.channels = 2
    cfg.data.window_seconds = 1.0
    cfg.train.device = "cpu"
    cfg.train.use_amp = False

    print("\n1. model builds from the released config")
    model = StreamingEMGCodec(cfg.model)
    n_params = count_parameters(model)
    check("parameter count matches the released checkpoint", n_params == 12_723_784,
          f"{n_params / 1e6:.3f} M")
    frame_rate = cfg.model.sample_rate / cfg.model.frame_size
    bits = cfg.model.rvq.num_codebooks * (cfg.model.rvq.codebook_size.bit_length() - 1)
    check("bitrate is 2400 bits/s/channel", frame_rate * bits == 2400,
          f"{frame_rate:.0f} Hz x {bits} bits")

    print("\n2. forward pass and loss")
    dataset = build_dataset(cfg.data, training=True)
    batch = torch.stack([dataset[i]["emg"] for i in range(2)])
    out = model(batch, n_codebooks=None)
    criterion = CodecLoss(cfg.loss)
    losses = criterion(out, batch)
    check("reconstruction keeps the input shape", out["reconstruction"].shape == batch.shape,
          str(tuple(out["reconstruction"].shape)))
    n_frames = batch.shape[-1] // cfg.model.frame_size
    check("one token vector per frame per channel",
          out["indices"].shape == (batch.shape[0], batch.shape[1], n_frames,
                                   cfg.model.rvq.num_codebooks),
          str(tuple(out["indices"].shape)))
    check("loss is finite", torch.isfinite(losses["loss"]).item(),
          f"loss={float(losses['loss'].detach()):.4f}")

    print("\n3. optimizer steps improve reconstruction")
    # The reconstruction terms are what the optimizer can improve immediately. The VQ and
    # commitment terms legitimately RISE early from a random codebook -- the encoder output
    # is still moving, so the quantization residual grows before the codebook catches up --
    # which is why this checks the reconstruction terms rather than the total.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        out = model(batch, n_codebooks=None)
        parts = criterion(out, batch)
        parts["loss"].backward()
        opt.step()
        recon = float(parts["huber"]) + float(parts["spectral"])
        first = recon if first is None else first
        last = recon
    check("huber + spectral decreased over 25 steps", last < first, f"{first:.4f} -> {last:.4f}")

    print("\n4. discriminator builds and scores both branches")
    disc = EMGDiscriminator(cfg.discriminator)
    scores, fmaps = disc(batch)
    check("one score per discriminator branch",
          len(scores) == len(cfg.discriminator.periods) + len(cfg.discriminator.stft_fft_sizes),
          f"{len(scores)} branches, {len(fmaps)} feature maps")

    print("\n5. checkpoint round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"model": model.state_dict(), "step": 8}, path)
        reloaded = StreamingEMGCodec(cfg.model)
        missing = reloaded.load_state_dict(torch.load(path, map_location="cpu")["model"],
                                           strict=True)
        with torch.no_grad():
            a = model(batch, n_codebooks=None)["reconstruction"]
            b = reloaded(batch, n_codebooks=None)["reconstruction"]
        check("reloaded model reproduces the output bit-for-bit",
              torch.equal(a, b), f"missing={list(missing.missing_keys)}")

    print("\n6. streaming path matches the offline forward")
    model.eval()
    frame = cfg.model.frame_size
    with torch.no_grad():
        offline = model(batch, n_codebooks=None)
        state = StreamingState()
        recs, toks = [], []
        for s in range(0, batch.shape[-1], frame):
            rec, tok, state = model.streaming_step(batch[..., s:s + frame], state=state,
                                                   n_codebooks=None)
            recs.append(rec)
            toks.append(tok)
        streamed = torch.cat(recs, dim=-1)
        tok_streamed = torch.cat(toks, dim=2)
    tok_match = float((offline["indices"] == tok_streamed).float().mean())
    rel = float((offline["reconstruction"] - streamed).norm()
                / offline["reconstruction"].norm().clamp_min(1e-8))
    check("tokens identical streamed vs offline", tok_match == 1.0, f"match={tok_match:.6f}")
    check("reconstruction identical streamed vs offline", rel < 1e-5, f"rel_l2={rel:.2e}")

    print("\n" + ("ALL PASSED" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

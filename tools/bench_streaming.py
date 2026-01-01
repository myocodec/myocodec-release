"""Streaming inference: correctness and cost.

    # is the fast path the same function as the reference?
    python tools/bench_streaming.py --config C --ckpt K --verify --val-root data/shards_val

    # what does a frame cost?
    python tools/bench_streaming.py --config C --ckpt K --bench

`--verify` replays a real held-out signal frame by frame through both paths and compares
tokens exactly. The claim being tested is equality, not closeness: the fast path reorders
the key axis of attention (permutation-invariant) and drops a mask that is all-true in the
steady state, so any disagreement is a bug rather than a rounding difference.

`--bench` reports per-frame latency and the real-time factor -- frame duration divided by
compute time -- for the reference path and for the session, at several batch sizes. Batch
here is streams x channels: the codec is channel-independent, so a 16-channel array is a
batch of 16.

`--split` additionally times the encoder and decoder halves apart, which is what you want
when the two run on different machines: the sensor encodes and ships 2400 bits/s/channel,
something else decodes. Each half is its own captured graph, so the two timings do not
share launch overhead and their sum is slightly above the fused `step()`.
"""
import argparse, glob, statistics, time

import numpy as np
import torch

from streaming_emg_codec.config import load_config
from streaming_emg_codec.model import StreamingEMGCodec
from streaming_emg_codec.model.codec import StreamingState
from streaming_emg_codec.model.fast_stream import StreamingSession


def load_model(cfg_path, ckpt, dev):
    cfg = load_config(cfg_path)
    model = StreamingEMGCodec(cfg.model).to(dev).eval()
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    return cfg, model


def real_frames(val_root, frame, n_frames, batch, dev):
    """Frames from a held-out shard, or noise if no shard set is given."""
    if not val_root:
        return torch.randn(n_frames, batch, 1, frame, device=dev)
    shards = sorted(glob.glob(f"{val_root}/shard_*.h5"))
    if not shards:
        raise SystemExit(f"no shards in {val_root}")
    import h5py
    with h5py.File(shards[0], "r") as f:
        x = np.asarray(f["windows"][:batch], dtype=np.float32)
    need = n_frames * frame
    if x.shape[1] < need:
        x = np.tile(x, (1, need // x.shape[1] + 1))
    x = torch.from_numpy(x[:, :need]).to(dev)                 # [B, n_frames*frame]
    return x.view(batch, n_frames, frame).permute(1, 0, 2).unsqueeze(2).contiguous()


@torch.no_grad()
def verify(cfg, model, args, dev):
    frame = cfg.model.frame_size
    for batch in args.batches:
        frames = real_frames(args.val_root, frame, args.frames, batch, dev)
        ref_state, ref_tok, ref_rec = StreamingState(), [], []
        for t in range(args.frames):
            r, i, ref_state = model.streaming_step(frames[t], state=ref_state, n_codebooks=None)
            ref_tok.append(i); ref_rec.append(r)
        ref_tok = torch.cat(ref_tok, dim=2); ref_rec = torch.cat(ref_rec, dim=-1)

        for label, kw in (("graph", dict(use_cuda_graph=True)),
                          ("eager", dict(use_cuda_graph=False))):
            sess = StreamingSession(model, batch_size=batch, device=dev,
                                    backend=args.backend,
                                    exact_mask=not args.fast_mask, **kw)
            tok, rec = [], []
            for t in range(args.frames):
                r, i = sess.step(frames[t])
                tok.append(i.clone()); rec.append(r.clone())
            tok = torch.cat(tok, dim=2); rec = torch.cat(rec, dim=-1)

            warm = sess.warmup_frames
            steady = slice(warm, args.frames)
            match_all = (tok == ref_tok).float().mean().item()
            match_steady = (tok[:, :, steady] == ref_tok[:, :, steady]).float().mean().item()
            rel = ((rec - ref_rec).norm() / ref_rec.norm()).item()
            status = "EXACT" if match_steady == 1.0 else "MISMATCH"
            print(f"  batch={batch:<3} {label:<5} tokens_all={match_all:.6f} "
                  f"tokens_steady={match_steady:.6f} recon_rel_l2={rel:.2e}  [{status}]")

        # the split entry points must agree with the fused one
        sess = StreamingSession(model, batch_size=batch, device=dev,
                                exact_mask=not args.fast_mask)
        tok, rec = [], []
        for t in range(args.frames):
            if sess.warmup_frames > t:
                r, i = sess.step(frames[t])
            else:
                i = sess.step_encode(frames[t])
                r = sess.step_decode(i)
            tok.append(i.clone()); rec.append(r.clone())
        tok = torch.cat(tok, dim=2); rec = torch.cat(rec, dim=-1)
        steady = slice(sess.warmup_frames, args.frames)
        m = (tok[:, :, steady] == ref_tok[:, :, steady]).float().mean().item()
        rel = ((rec - ref_rec).norm() / ref_rec.norm()).item()
        print(f"  batch={batch:<3} split tokens_steady={m:.6f} recon_rel_l2={rel:.2e}  "
              f"[{'EXACT' if m == 1.0 else 'MISMATCH'}]")


def timed(fn, n, warm, dev):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize() if dev.type == "cuda" else None
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return statistics.mean(ts), ts[len(ts) // 2], ts[int(len(ts) * 0.99)]


@torch.no_grad()
def bench_split(cfg, model, args, dev):
    """Encoder and decoder latency, measured apart."""
    frame = cfg.model.frame_size
    frame_ms = 1000.0 * frame / cfg.model.sample_rate
    print(f"{'batch':>5} {'half':>8} {'path':>18} {'ms/frame':>9} {'p50':>8} {'p99':>8} {'RTF':>9}")
    print("-" * 74)
    for batch in args.batches:
        x = torch.randn(batch, 1, frame, device=dev)
        # reference halves, each with its own cache
        ec = model.encoder.allocate_inference_cache(batch, dev)
        dc = model.decoder.allocate_inference_cache(batch, dev)
        idx = None
        for _ in range(64):
            idx, ec = model.encode(x, None, ec)
            _, dc = model.decode(idx, None, dc)
        rows = []
        def ref_enc():
            nonlocal ec
            _, ec = model.encode(x, None, ec)
        def ref_dec():
            nonlocal dc
            _, dc = model.decode(idx, None, dc)
        rows.append(("encode", "reference", timed(ref_enc, args.iters, args.warmup, dev)))
        rows.append(("decode", "reference", timed(ref_dec, args.iters, args.warmup, dev)))

        for tag, compiled in (("session", False), ("session+compile", True)):
            if compiled and not args.compile:
                continue
            sess = StreamingSession(model, batch_size=batch, device=dev, backend=args.backend,
                                    compile=compiled, exact_mask=not args.fast_mask)
            for _ in range(sess.warmup_frames):
                sess.step(x)
            codes = sess.step_encode(x)
            for _ in range(8):
                codes = sess.step_encode(x)
                sess.step_decode(codes)
            rows.append(("encode", tag, timed(lambda: sess.step_encode(x), args.iters, args.warmup, dev)))
            rows.append(("decode", tag, timed(lambda: sess.step_decode(codes), args.iters, args.warmup, dev)))

        for half, tag, (mean, p50, p99) in rows:
            print(f"{batch:>5} {half:>8} {tag:>18} {mean:>9.3f} {p50:>8.3f} {p99:>8.3f} "
                  f"{frame_ms / mean:>9.1f}")
        print()


@torch.no_grad()
def bench(cfg, model, args, dev):
    frame = cfg.model.frame_size
    frame_ms = 1000.0 * frame / cfg.model.sample_rate
    print(f"device = {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}")
    print(f"frame  = {frame} samples = {frame_ms:.1f} ms\n")
    print(f"{'batch':>5} {'path':>24} {'mean ms':>9} {'p50':>8} {'p99':>8} "
          f"{'RTF':>9} {'streams':>8} {'speedup':>8}")
    print("-" * 88)
    for batch in args.batches:
        x = torch.randn(batch, 1, frame, device=dev)
        base = None
        labels = ["reference", "session (eager)", "session (graph)"]
        if args.compile:
            labels.append("session (graph+compile)")
        for label in labels:
            if label == "reference":
                st = StreamingState()
                for _ in range(64):
                    model.streaming_step(x, state=st, n_codebooks=None)
                fn = lambda: model.streaming_step(x, state=st, n_codebooks=None)  # noqa: E731
            else:
                sess = StreamingSession(model, batch_size=batch, device=dev,
                                        use_cuda_graph="graph" in label,
                                        compile="compile" in label,
                                        backend=args.backend,
                                        exact_mask=not args.fast_mask)
                for _ in range(sess.warmup_frames + 8):
                    sess.step(x)
                fn = lambda s=sess: s.step(x)  # noqa: E731
            mean, p50, p99 = timed(fn, args.iters, args.warmup, dev)
            base = mean if base is None else base
            rtf = frame_ms / mean
            print(f"{batch:>5} {label:>24} {mean:>9.3f} {p50:>8.3f} {p99:>8.3f} "
                  f"{rtf:>9.1f} {rtf * batch:>8.0f} {base / mean:>7.1f}x")
        print()
    if dev.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-root", default=None, help="held-out shards, for --verify")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--frames", type=int, default=200, help="frames per --verify stream")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--fast-mask", action="store_true",
                    help="drop the all-true attention mask (may cost token exactness)")
    ap.add_argument("--compile", action="store_true", help="also bench the torch.compile path")
    ap.add_argument("--backend", default="auto", choices=["auto", "flash", "sdpa"])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--split", action="store_true", help="time the encoder and decoder halves apart")
    a = ap.parse_args()
    if not (a.verify or a.bench or a.split):
        a.verify = a.bench = True

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = load_model(a.config, a.ckpt, dev)
    if a.verify:
        print("=== correctness: fast path vs reference path ===")
        verify(cfg, model, a, dev)
        print()
    if a.bench:
        print("=== cost per frame ===")
        bench(cfg, model, a, dev)
        print()
    if a.split:
        print("=== cost per frame, halves measured separately ===")
        bench_split(cfg, model, a, dev)


if __name__ == "__main__":
    main()

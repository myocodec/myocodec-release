"""Extract windows with PER-WINDOW, ACROSS-CHANNEL normalization, applying EACH DATASET'S
OWN documented preprocessing first (streaming_emg_codec.data.conditioning.PAPER_SPEC).

Why here and not later: the shards carry no dataset label -- they are a bare `windows`
array with no attrs -- so conditioning cannot be retrofitted to them. The manifest row does
carry `dataset`, so extraction is the only point where per-dataset filtering is possible.

It is also the CORRECT point. Filtering after ninapro's (stimulus, repetition) segmentation
would put 14-35% of every 200-506 sample segment inside the filter transient; here the whole
recording is filtered before it is cut into windows.

Cutoffs are in Hz so applying them at the common 2 kHz rate is equivalent to applying them
at each dataset's native rate, over the band that survives resampling.

Only 17.87% of the corpus is touched: gaddy (its own butter(3,2Hz)+60Hz notch comb), hyser
(BP 10-500 + 50 Hz comb, since we read the *raw* files), csl (BP 20-400), emgepn and
emg2speech (50 Hz notch on request). The rest already ship their paper's preprocessing.
"""
import json, os, random, argparse, sys, numpy as np, h5py
from streaming_emg_codec.data.conditioning import condition_paper, PAPER_SPEC, TRIM_MS
C = os.environ.get("EMG_CORPUS_ROOT", "data/corpus2k").rstrip("/") + "/"
SH = os.environ.get("EMG_SHARD_ROOT", "data/shards").rstrip("/") + "/"
FS = 2000.0
W = 10000                # samples per window: 5.0 s at 2 kHz
SHARD_WINDOWS = 50000   # ~1GB fp16/shard; ~2GB fp32 buffer/worker (safe for many parallel workers)

def run(target, split, worker, nworkers):
    rows = [json.loads(l) for l in open(C + "manifest.jsonl") if l.strip()]
    rows = [r for r in rows if r["split"] == split]
    w = np.array([r["n_ch"] * r["n_samp"] for r in rows], dtype=np.float64)
    scale = max(w.sum() / max(target, 1), 1e-9); wpr = np.maximum(1, np.round(w / scale)).astype(np.int64)
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["h5"], rows[i]["key"]))[worker::nworkers]
    os.makedirs(SH, exist_ok=True); rnd = random.Random(worker); h5c = {}; buf = []; si = [0]; tot = [0]; nfilt = [0]
    def flush():
        if not buf: return
        rnd.shuffle(buf); a = np.stack(buf).astype(np.float16)
        with h5py.File(f"{SH}shard_w{worker}_{si[0]:03d}.h5", "w") as f:
            f.create_dataset("windows", data=a, chunks=(min(4096, len(a)), W))
        si[0] += 1; tot[0] += len(buf); buf.clear()
        print(f"w{worker}: {tot[0]} windows, {si[0]} shards", flush=True)
    for i in order:
        r = rows[i]; h5 = h5c.get(r["h5"]) or h5c.setdefault(r["h5"], h5py.File(r["h5"], "r"))
        arr = np.asarray(h5["rec"][r["key"]][:], dtype=np.float32)
        spec = PAPER_SPEC.get(r["dataset"])
        if spec and spec.get("act"):
            # filter the WHOLE recording, then drop the causal transient, then window
            arr, _d = condition_paper(arr, FS, r["dataset"], trim_ms=TRIM_MS)
            arr = np.asarray(arr, dtype=np.float32)
            nfilt[0] += 1
        if arr.shape[1] < 64:
            continue
        Cc, T = arr.shape
        for _ in range(int(wpr[i])):
            s0 = rnd.randrange(0, max(1, T - W + 1)) if T > W else 0
            win = arr[:, s0:s0 + W]                 # ALL channels of this window -> across-channel stats
            m = win.mean(); sd = win.std()          # shared per-window mean/std across channels+time
            ch = rnd.randrange(Cc)
            x = arr[ch, s0:s0 + W].astype(np.float32)
            if x.shape[0] < W: x = np.pad(x, (0, W - x.shape[0]))
            buf.append((x - m) / sd if sd > 1e-5 else x - m)
            if len(buf) >= SHARD_WINDOWS: flush()
    flush(); print(f"DONE worker {worker}: {tot[0]} windows in {si[0]} shards, {nfilt[0]} recordings filtered", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=int, default=8000000)
    ap.add_argument("--split", default="train"); ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args(); run(a.target, a.split, a.worker, a.nworkers)

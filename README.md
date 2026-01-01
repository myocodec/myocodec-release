# StreEMG — a streaming neural codec for surface EMG

Reference implementation for *StreEMG: Streaming Neural EMG Codec*.

A causal transformer encoder, residual vector quantizer, and causal transformer decoder
that encode **each sEMG channel independently** at a **50 Hz frame rate** and
**2400 bits/s/channel** (6 codebooks × 8 bits × 50 Hz). The model is channel-count
agnostic: channels fold into the batch dimension, so a codec trained on 1-channel windows
runs unchanged on 16- or 256-channel arrays.

**This repository covers codec pretraining only** — corpus preparation, the two-stage
training run, and reconstruction evaluation. The downstream benchmarks reported in the
paper are separate codebases and are not included here.

## Contents

```
configs/          pretrain_stage1.yaml, pretrain_stage2.yaml   the released run, exactly
pyproject.toml    optional: pip install -e .
streaming_emg_codec/
  config.py       typed config; unknown YAML keys are an error, not a silent default
  train.py        two-stage trainer (Muon/AdamW split, AMP, disc ramp, resume)
  losses.py       Huber + multi-scale spectral + VQ + commitment
  model/
    codec.py      encoder / RVQ / decoder, framing, and the streaming interface
    attention.py  sliding-window causal attention with a bounded KV cache
    rvq.py        residual VQ with dead-entry restart
    discriminator.py   multi-period + multi-scale-STFT discriminator
    fast_stream.py  low-latency inference: static KV buffers + one CUDA graph
  data/
    readers.py       per-format readers, one per source corpus
    build_corpus.py  raw corpora -> unified 2 kHz HDF5 + manifest, with splits
    conditioning.py  per-corpus preprocessing table (PAPER_SPEC) and causal filters
    extract_shards.py  conditioned, normalized 5 s windows -> h5 shards
    shards.py        the training dataset (block-wise mmap streamer)
tools/
  eval_recon.py       reconstruction metrics; the script behind every number in Table 5
  verify_streaming.py streamed vs offline equivalence on a trained checkpoint
  bench_streaming.py  fast-path correctness (--verify) and per-frame cost (--bench)
tests/
  test_smoke.py           end-to-end, no data and no GPU needed
  test_streaming_cache.py attention cache: equivalence, bounded memory, no lookahead
docs/DATA.md        corpus composition, splits, and per-corpus preprocessing
```

## Install

```bash
pip install -r requirements.txt
```

Every command below runs from the repo root with no further setup — the tools and tests
bootstrap the package onto `sys.path` themselves, so `PYTHONPATH` is not needed. To install
it into an environment instead, `pip install -e .` (extras: `.[train]` pulls Muon and
wandb, `.[data]` the corpus readers).

`flash-attn` is optional for training and evaluation: without it the attention layer falls
back to `torch` SDPA, which applies the same `window_size=(W, 0)` sliding-window causal
mask. Attention runs over frames (100–250 tokens for a 5 s window), not raw samples, so the
fallback costs little. The two are not bit-identical — they reduce in a different order, so
~0.1% of tokens can land on the other side of a quantizer boundary — which matters only if
you are comparing token streams across machines. Install it if you can, and see
[Streaming inference](#streaming-inference) for where it also buys exactness.

## Quick check

```bash
python tests/test_smoke.py
```

Builds the paper's model from `configs/pretrain_stage1.yaml`, runs a few optimizer steps on
synthetic windows, round-trips a checkpoint, and confirms the streaming path reproduces the
offline forward. Runs on CPU in under a minute. It asserts the parameter count is
**12,723,784**, so it also tells you the config in this repo is the one the released
checkpoint was trained with.

## Data

See [`docs/DATA.md`](docs/DATA.md) — twelve public sEMG corpora, 25,114 channel-hours, each
preprocessed with its own published pipeline, and how to rebuild the window shards.

## Train

Two stages that differ **only in batch size and step budget**. Architecture, losses,
optimizer and discriminator schedule are identical between the two config files, so the
transition changes the objective and nothing else.

```bash
# Stage 1 — generator only, batch 1024, to step 100k
scripts/train.sh --config configs/pretrain_stage1.yaml --resume auto \
  --override data.root=$EMG_SHARD_ROOT --override train.output_dir=checkpoints/streemg

# Stage 2 — discriminator ramps in over 50k steps, batch 192
scripts/train.sh --config configs/pretrain_stage2.yaml --resume auto \
  --override data.root=$EMG_SHARD_ROOT --override train.output_dir=checkpoints/streemg
```

`scripts/slurm_example.sh` is a self-resubmitting SLURM chain that picks the stage from the
checkpoint's own step. Four things are worth knowing before starting a long run:

- **Step count is a misleading proxy for data exposure.** Stage 1 runs at a 5.3× larger
  batch, so the majority of the data the codec ever sees is seen before step 100k. The
  released checkpoint is at step 200,000: 48% of its steps but 83% of its data came from
  stage 1, and it had already completed 10 full epochs of the 9,593,476-window corpus by
  the end of stage 1.
- **Never resume a post-100k checkpoint with the stage-1 config.** The discriminator engages
  at `start_step >= 100000` and immediately OOMs at stage 1's batch of 1024. The SLURM
  script reads the stage from the checkpoint to make that unrepresentable.
- **`train.max_gpu_memory_gb` is a hard cap, not a hint.** The trainer raises when the
  allocator crosses it rather than growing into another process on a shared card. Set it to
  what you actually want to occupy, and note that a too-low cap makes a long run
  crash-loop rather than fail loudly: our stage 1 stopped at step 95,276 for this reason,
  and stage 2 resumed from there.
- **RVQ results are not reproducible without pinning the checkpoint.** Codebook occupancy
  drifts across saves; quote a specific `step_*.pt`, not "the final model".

## Evaluate

```bash
# reconstruction metrics on a held-out shard set
python tools/eval_recon.py --model streemg \
  --config configs/pretrain_stage2.yaml --ckpt checkpoints/streemg/step_200000.pt \
  --val-root $EMG_SHARD_ROOT_VAL --tag streemg-200k

# streaming equivalence: chunk-by-chunk cached decode == full-sequence forward
python tools/verify_streaming.py \
  --config configs/pretrain_stage2.yaml --ckpt checkpoints/streemg/step_200000.pt \
  --val-root $EMG_SHARD_ROOT_VAL --tag streemg-200k
```

`eval_recon.py` also has a `biocodec` backend so both rows of the paper's reconstruction
table come from identical metric code. That backend needs the baseline's own released
repository and checkpoint, which are not redistributed here; pass `--biocodec-repo`.

## Streaming inference

`model.streaming_step()` is the reference implementation and is what the tests check
against, but it is not how you would deploy this. At one frame per step the codec is
**launch-bound, not compute-bound**: a frame costs ~6.4 ms on an RTX PRO 6000, and 64
channels batched together cost 7.2 ms — 64× the arithmetic for 13% more time. The 6.4 ms
is several hundred tiny CUDA kernels plus three `.item()` calls per frame in the
quantizer's variable-bitrate bookkeeping.

`StreamingSession` keeps the model and its weights exactly as they are and removes that
overhead — no host synchronisation, static KV buffers, and the whole encode+decode step
captured into one CUDA graph:

```python
from streaming_emg_codec.model.fast_stream import StreamingSession

session = StreamingSession(model, batch_size=1, device="cuda")
for frame in stream:                       # [B, C, 40] at 2 kHz
    recon, codes = session.step(frame)
```

Measured on one RTX PRO 6000 Blackwell, fp32 weights, bf16 KV cache, batch = streams ×
channels (`tools/bench_streaming.py --bench`).

> **Measurement conditions, and which way each number is biased.** The benchmark card was
> idle for compute but shared a host with a second card training other jobs, and it idles
> at 180 MHz against a 2430 MHz maximum, so the clock may not have fully ramped within the
> warmup. Both effects only ever add time. So:
>
> * **absolute ms/frame is an upper bound** and **RTF a lower bound** — an idle, fully
>   boosted card is at least this good;
> * **the speedup ratios are not conservative and may be inflated.** The reference path is
>   launch-bound, several hundred kernel launches per frame against the session's one, so
>   host-side contention taxes it far harder than it taxes the session. On a quiet machine
>   the gap can narrow even as both absolute numbers improve. Trust the latency budget;
>   treat every `N×` below as provisional.
>
> We report the pessimistic measurement rather than wait for an empty machine, because the
> claim the absolute numbers support — far faster than real time — survives it. To get the
> tight numbers, re-run `--bench --split` on an idle card and record
> `nvidia-smi --query-gpu=clocks.sm` alongside: if the clock reaches its maximum during the
> run, the ramp concern is gone and only host load remains.

| batch | path | ms/frame | p99 | RTF | real-time streams |
|---:|---|---:|---:|---:|---:|
| 1 | reference | 6.356 | 10.412 | 3.1 | 3 |
| 1 | session | 0.774 | 0.786 | **25.8** | 26 |
| 1 | session + `compile=True` | **0.470** | 0.481 | **42.6** | 43 |
| 16 | reference | 6.903 | 8.189 | 2.9 | 46 |
| 16 | session | 1.122 | 1.160 | 17.8 | 285 |
| 16 | session + `compile=True` | 0.746 | 0.775 | 26.8 | 429 |
| 64 | reference | 7.196 | 13.988 | 2.8 | 178 |
| 64 | session | 1.371 | 1.478 | 14.6 | 934 |
| 64 | session + `compile=True` | 1.006 | 1.113 | 19.9 | **1272** |

### Encoder and decoder separately

The two halves are independently useful — a sensor encodes and ships 2400 bits/s/channel,
something else decodes — so `step_encode()` and `step_decode()` drive them apart, each with
its own captured graph and its own stream position:

| batch | half | reference | session | session + compile | RTF (compile) |
|---:|---|---:|---:|---:|---:|
| 1 | encode | 3.505 | 0.437 | **0.265** | **75.6** |
| 1 | decode | 2.999 | 0.362 | **0.242** | **82.6** |
| 16 | encode | 3.958 | 0.591 | 0.399 | 50.1 |
| 16 | decode | 3.382 | 0.545 | 0.367 | 54.4 |
| 64 | encode | 3.838 | 0.713 | 0.510 | 39.2 |
| 64 | decode | 3.373 | 0.648 | 0.484 | 41.3 |

(ms per frame, same conditions and the same caveats as above — absolute times are upper
bounds, the **13.2× encode / 12.4× decode** at batch 1 are provisional.) The halves are near-symmetric,
which is what the parameter counts predict — 6.34 M each side. Their sum, 0.507 ms, is
slightly above the fused `step()` at 0.470 ms: two graph launches instead of one. Splitting
costs about 8%, so fuse when both halves run on the same device.

**8.2× at batch 1 fused, 13.5× with `compile=True`** (provisional — see the bias note above). Tail latency matters more than the mean for
a streaming codec, and it improves by more: p99 falls from 10.4 ms to 0.48 ms, because the
variance was host-side scheduling rather than GPU work.

**The fast path is bitwise identical to the reference** — not approximately, not
token-exact, but `recon_rel_l2 == 0.0` at every batch size tested. Two things make that
true. The steady-state key set is exactly `W+1` keys, all of which pass both the causal and
the sliding-window mask, so no mask is needed and the buffer can be a ring; and the session
dispatches `flash_attn_with_kvcache`, the same kernel the reference uses, rather than a
different attention implementation. The first `W+1 = 65` frames genuinely have a different
key set, so they run on the reference path verbatim and the cache is seeded from it. Check
it yourself:

```bash
python tools/bench_streaming.py --config configs/pretrain_stage2.yaml \
  --ckpt streemg_step200000_model.pt --verify --val-root $EMG_SHARD_ROOT_VAL
```

`--verify` covers the fused path, the un-captured path and the split entry points; all
three come out bitwise identical. `--bench` and `--split` produce the two tables above.

Two settings worth knowing:

- **`backend`** is `"flash"` when flash-attn is importable and `"sdpa"` otherwise. Only the
  flash backend is bitwise identical to a flash-attn reference. `sdpa` is portable and, with
  `compile=True`, the fastest configuration here — **0.359 ms, RTF 55.6, 18.0×** at batch 1 —
  because Inductor can fuse through SDPA but not through flash-attn's opaque custom op. The
  catch is that the two reduce in a different order, so ~0.1% of tokens land on the other
  side of a quantizer boundary. Against an SDPA reference (no flash-attn installed) the
  `sdpa` backend is token-exact at batch 1. Pick `flash` for reproducible token streams and
  `sdpa` + `compile` for throughput.
- **`compile=True`** adds a one-off Inductor compile pause of a minute or two on the first
  steady-state frame, and its fused kernels are not the reference's. It is off by default
  for that reason.

Peak memory is roughly 2× the reference (227 MB vs 118 MB at batch 1; 5.8 GB vs 3.0 GB at
batch 64), which is the CUDA graph's private pool holding the captured intermediates.

## Checkpoint

`step_200000.pt` is the checkpoint every StreEMG number in the paper uses.

| file | contents | resumable |
|---|---|---|
| `streemg_step200000_model.pt` | generator weights only (12.72 M params) | no |
| `streemg_step200000_full.pt` | generator + discriminator + both optimizer states | yes |

Load the model-only file with:

```python
import torch
from streaming_emg_codec.config import load_config
from streaming_emg_codec.model import StreamingEMGCodec

cfg = load_config("configs/pretrain_stage2.yaml")
model = StreamingEMGCodec(cfg.model)
model.load_state_dict(torch.load("streemg_step200000_model.pt", map_location="cpu")["model"])
model.eval()

# x: [batch, channels, samples] at 2 kHz, normalized per window across channels
out = model(x)                      # out["reconstruction"], out["indices"]
```

For online use, `model.streaming_step(chunk, state)` consumes exactly one 40-sample frame
at a time against a bounded KV cache and produces bit-identical tokens to the offline
forward — `tests/test_streaming_cache.py` checks equivalence, that the cache stays bounded
over 8000 frames, and that perturbing frame *t* leaves every earlier output untouched.

## Notes on reproduction

- **Normalization is per-window and across-channel**: one mean and one standard deviation
  shared over channels and time. Per-channel normalization erases relative amplitude
  between channels, which the codec is meant to preserve.
- **Input rate is 2 kHz.** Resample anything else, anti-aliased, before encoding.
- **Windows are 5.0 s** (10,000 samples at 2 kHz) throughout training and evaluation.
- The optimizer is **Muon on 2-D hidden matrices, AdamW on everything else** — the RVQ
  codebooks are lookup tables, not linear maps, and the input/output projections touch the
  data boundary, so both stay on AdamW along with all 1-D parameters.
- **The token stream is precision-sensitive.** Streaming and offline encoding agree on
  99.98% of frames in fp32, but only 95.7% under the bfloat16 autocast used for training.
  This is not a cache defect: RVQ assigns each frame by nearest neighbour, so a ~1e-4 bf16
  difference flips frames near a Voronoi boundary, and the residual structure compounds the
  flip down the codebooks (per-codebook agreement falls monotonically from 1.000 to 0.926).
  Reconstruction is barely affected — the flipped frames are near-equidistant between
  entries — but if you need a frame-for-frame reproducible token stream, encode in fp32.
  `tools/verify_streaming.py --amp` reports both. This is a property of the *offline vs
  streaming* comparison; the fast inference path of `StreamingSession` is bitwise identical
  to the reference streaming path, so it inherits this and adds nothing to it.

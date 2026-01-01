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
  test_verify_catches_divergence.py  negative control: break the fast path, require the
                          equivalence check to notice (CUDA; correctness only)
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
**launch-bound, not compute-bound**: a frame costs ~5.6 ms on an RTX PRO 6000, and 64
channels batched together cost 6.3 ms — 64× the arithmetic for 14% more time. The 5.6 ms
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

> **Measurement conditions.** Idle card, nothing else running: verified `0` training
> processes, `0` CUDA contexts and `0%` utilisation immediately before the run. Cold
> Inductor cache, so the compiled kernels were autotuned on the same quiet machine they
> were then measured on. SM clock sampled every 0.5 s *during* the timed loops: median
> 2355 MHz against a 2430 MHz maximum, i.e. 97% — the card is fully ramped and clock state
> contributes at most ~3%.
>
> Run-to-run spread across repeated idle runs is about 1% on the fused step and about 0.5%
> on the halves, measured across independent single-session processes.
>
> **The speedup column is the soft number.** The reference path is launch-bound — several
> hundred kernel launches per frame against the session's one — so host contention taxes it
> far harder than it taxes the session, and any measurement on a busy machine inflates the
> ratio. We measured this rather than assuming it: on a contended host the same code read
> 13.5× at batch 1, and on this idle one it reads 12.0×. The absolute latencies moved
> hardly at all (the session path by 0.6%); it was the reference that got faster. Quote the
> latency budget; treat the ratio as approximate.

| batch | path | ms/frame | p99 | RTF | real-time streams |
|---:|---|---:|---:|---:|---:|
| 1 | reference | 5.585 | 5.975 | 3.6 | 4 |
| 1 | session | 0.774 | 0.784 | 25.8 | 26 |
| 1 | session + `compile=True` | **0.467** | 0.477 | **42.8** | 43 |
| 16 | reference | 6.175 | 6.520 | 3.2 | 52 |
| 16 | session | 1.124 | 1.159 | 17.8 | 285 |
| 16 | session + `compile=True` | 0.666 | 0.705 | 30.0 | 481 |
| 64 | reference | 6.343 | 6.661 | 3.2 | 202 |
| 64 | session | 1.380 | 1.490 | 14.5 | 927 |
| 64 | session + `compile=True` | 1.013 | 1.118 | 19.7 | **1264** |

**7.2× at batch 1 fused, 12.0× with `compile=True`.** Tail latency matters more than the
mean for a streaming codec, and it improves by far more: p99 falls from 5.98 ms to 0.48 ms,
because most of the variance was host-side scheduling rather than GPU work.

### Encoder and decoder separately

The two halves are independently useful — a sensor encodes and ships 2400 bits/s/channel,
something else decodes — so `step_encode()` and `step_decode()` drive them apart, each with
its own captured graph and its own stream position:

| batch | half | reference | session | session + compile | RTF (compile) | p99 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | encode | 3.060 | 0.433 | **0.253** | **79.1** | 0.265 |
| 1 | decode | 2.650 | 0.359 | **0.229** | **87.3** | 0.243 |
| 16 | encode | 3.385 | 0.589 | 0.350 | 57.1 | 0.360 |
| 16 | decode | 2.927 | 0.544 | 0.322 | 62.1 | 0.334 |
| 64 | encode | 3.380 | 0.711 | 0.507 | 39.4 | 0.587 |
| 64 | decode | 2.950 | 0.646 | 0.469 | 42.6 | 0.547 |

> The batch-16 and batch-64 rows were taken before a defect in the benchmark harness was
> found and are biased **upward by a few percent**; they are being re-measured. A
> `StreamingSession` holds a private CUDA graph memory pool for its lifetime, and the
> harness used to time several configurations in one process without releasing them, so
> whichever measurement ran last was inflated — 4.5% at batch 1, and 13% in a case holding
> more sessions. The batch-1 row above is clean: it comes from single-session-per-process
> runs, reproduced to three digits by two independent protocols. `tools/bench_streaming.py`
> now frees each session between configurations; see `_release` there.

(ms per frame, same conditions as above; **12.1× encode, 11.6× decode** at batch 1.) The halves are near-symmetric,
which is what the parameter counts predict — 6.34 M each side. Splitting is close to free: an
alternating encode-then-decode pair measures **0.467 ms against 0.464 ms fused, about 1%**.

Do not estimate that cost by adding the two half-timings above — they are measured in
isolation, each in its own tight loop, so the sum double-counts per-call overhead that the
alternating pattern amortises. The sum says 0.482 ms and would have you believe splitting
costs 4%; measured back to back it costs 1%.

Read the 1% as an **upper bound**. Both sides of it were timed in processes holding three
captured graphs, identically, so the comparison is sound — but a per-replay overhead does
not cancel in a ratio when one side replays once and the other twice, which biases it
upward. The true figure is at or below 1%; a minimal-residency measurement, one graph per
process, is pending.

And note what the 1% is a cost *of*: two captured graphs against one, on one card in one
process. It prices the extra graph launch and nothing else. Splitting the **model** is
nearly free; splitting a **deployment** additionally pays whatever link sits between the
sensor and the decoder, which this does not measure.

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
three come out bitwise identical. And because a check that has only ever passed is not by
itself evidence, `tests/test_verify_catches_divergence.py` is its negative control: it
breaks the fast path four ways — stream position off by one, KV length off by one, a wiped
layer-0 cache, a perturbed encoder weight — and requires the comparison to fail on every
one. It also confirms the comparison set is non-empty and the two outputs are distinct
tensors, which is how such a check usually fails silently. `--bench` and `--split` produce the two tables above.

Two settings worth knowing:

- **`backend`** is `"flash"` when flash-attn is importable and `"sdpa"` otherwise. Only the
  flash backend is bitwise identical to a flash-attn reference. `sdpa` is portable and, with
  `compile=True`, the fastest configuration here — **0.359 ms, RTF 55.6** at batch 1 (measured on a busy host; not re-run idle) —
  because Inductor can fuse through SDPA but not through flash-attn's opaque custom op. The
  catch is that the two reduce in a different order, so ~0.1% of tokens land on the other
  side of a quantizer boundary. Against an SDPA reference (no flash-attn installed) the
  `sdpa` backend is token-exact at batch 1. Pick `flash` for reproducible token streams and
  `sdpa` + `compile` for throughput.
- **`compile=True`** adds a one-off Inductor compile pause of a minute or two on the first
  steady-state frame, and its fused kernels are not the reference's. It is off by default
  for that reason.

Peak memory at batch 1, with the weights resident (50.9 MB of each total). Each
configuration measured in its own process, so no other session's graph pool is counted:

| path | steady state | including one-off setup |
|---|---:|---:|
| reference | 118 MB | 118 MB |
| session | 194 MB | 194 MB |
| session + `compile=True` | 194 MB | 194 MB, or 295 MB on a first compile |

The ~76 MB over the reference is the CUDA graph's private pool holding the captured
intermediates, and compiling adds nothing to it — Inductor fuses kernels but the pool holds
a similar set of intermediates either way. The one exception is a *first* compile, where
Inductor's autotuning transiently needs about 100 MB more; once its cache is warm that
disappears, so a deployment pays it once per machine, not per process. At batch 64 the
session needs about 5.8 GB against the reference's 3.0 GB.

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

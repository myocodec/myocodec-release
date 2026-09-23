# MyoCodec: A Streaming Neural Codec for Electromyography

Code for the ICLR 2027 submission *MyoCodec: A Streaming Neural Codec for
Electromyography*.

A causal transformer encoder, residual vector quantizer, and causal transformer decoder
that encode **each sEMG channel independently** at a **50 Hz frame rate** and
**2400 bits/s/channel** (6 codebooks × 8 bits × 50 Hz). Channels fold into the batch
dimension, so a codec trained on 1-channel windows runs unchanged on 16- or 256-channel
arrays.

**This repository covers codec pretraining only** — corpus preparation, the two-stage
training run, and reconstruction evaluation. The downstream benchmarks reported in the
paper are separate codebases and are not included here.

## Contents

```
configs/          pretrain_stage1.yaml, pretrain_stage2.yaml   the released run, exactly
scripts/train.sh  launcher for the two training stages
pyproject.toml    optional: pip install -e .
streaming_emg_codec/
  config.py       typed config; unknown YAML keys are an error, not a silent default
  train.py        two-stage trainer (Muon/AdamW split, AMP, disc ramp, resume)
  losses.py       Huber + multi-scale spectral + VQ + commitment
  model/
    codec.py          encoder / RVQ / decoder, framing, and the streaming interface
    attention.py      sliding-window causal attention with a bounded KV cache
    rvq.py            residual VQ with dead-entry restart
    discriminator.py  multi-period + multi-scale-STFT discriminator
    fast_stream.py    low-latency inference: static KV buffers + one CUDA graph
  data/
    readers.py        per-format readers, one per source corpus
    build_corpus.py   raw corpora -> unified 2 kHz HDF5 + manifest, with splits
    conditioning.py   per-corpus preprocessing table (PAPER_SPEC) and causal filters
    extract_shards.py conditioned, normalized 5 s windows -> h5 shards
    shards.py         the training dataset (block-wise mmap streamer)
tools/
  eval_recon.py       reconstruction metrics
  verify_streaming.py streamed vs offline equivalence on a trained checkpoint
docs/DATA.md        corpus composition, splits, and per-corpus preprocessing
```

## Install

```bash
pip install -r requirements.txt
```

Every command below runs from the repo root with no further setup — the tools bootstrap the
package onto `sys.path` themselves, so `PYTHONPATH` is not needed. To install into an
environment instead, `pip install -e .` (extras: `.[train]` pulls Muon and wandb, `.[data]`
the corpus readers).

`flash-attn` is optional. Without it the attention layer falls back to `torch` SDPA, which
applies the same `window_size=(W, 0)` sliding-window causal mask. The two are not
bit-identical — they reduce in a different order, so ~0.1% of tokens can land on the other
side of a quantizer boundary — which matters only if you are comparing token streams across
machines.

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
  --override data.root=$EMG_SHARD_ROOT --override train.output_dir=checkpoints/myocodec

# Stage 2 — discriminator ramps in over 50k steps, batch 192
scripts/train.sh --config configs/pretrain_stage2.yaml --resume auto \
  --override data.root=$EMG_SHARD_ROOT --override train.output_dir=checkpoints/myocodec
```

Two things will break a long run if you do not know them:

- **Never resume a post-100k checkpoint with the stage-1 config.** The discriminator
  engages at `start_step >= 100000` and immediately OOMs at stage 1's batch of 1024.
- **`train.max_gpu_memory_gb` is a hard cap, not a hint.** The trainer raises when the
  allocator crosses it rather than growing into another process on a shared card. Set it to
  what you actually want to occupy, and note that a too-low cap makes a long run crash-loop
  rather than fail loudly.

## Evaluate

```bash
# reconstruction metrics on a held-out shard set
python tools/eval_recon.py --model myocodec \
  --config configs/pretrain_stage2.yaml --ckpt checkpoints/myocodec/step_200000.pt \
  --val-root $EMG_SHARD_ROOT_VAL --tag myocodec-200k

# streaming equivalence: chunk-by-chunk cached decode == full-sequence forward
python tools/verify_streaming.py \
  --config configs/pretrain_stage2.yaml --ckpt checkpoints/myocodec/step_200000.pt \
  --val-root $EMG_SHARD_ROOT_VAL --tag myocodec-200k
```

`eval_recon.py` also has a `biocodec` backend, so both rows of the paper's reconstruction
table come from identical metric code. That backend needs the baseline's own released
repository and checkpoint, which are not redistributed here; pass `--biocodec-repo`.

## Streaming inference

`model.streaming_step(chunk, state)` is the reference implementation: it consumes exactly
one 40-sample frame at a time against a bounded KV cache. `StreamingSession` runs the same
weights with static KV buffers and the encode+decode step captured into one CUDA graph:

```python
from streaming_emg_codec.model.fast_stream import StreamingSession

session = StreamingSession(model, batch_size=1, device="cuda")
for frame in stream:                       # [B, C, 40] at 2 kHz
    recon, codes = session.step(frame)
```

`step_encode(frame)` and `step_decode(indices)` drive the two halves independently, each
with its own captured graph and its own stream position — for the case where a sensor
encodes and ships 2400 bits/s/channel and something else decodes.

The first `W + 1 = 65` frames genuinely have a different key set from the steady state, so
they run on the reference path verbatim and the cache is seeded from them. Two constructor
settings change behaviour:

- **`backend`** is `"flash"` when flash-attn is importable and `"sdpa"` otherwise. The flash
  backend dispatches `flash_attn_with_kvcache`, the same kernel the reference uses, and is
  bitwise identical to it. `sdpa` is the portable fallback and reduces in a different order,
  so ~0.1% of tokens can land on the other side of a quantizer boundary.
- **`compile=True`** runs the step through `torch.compile` before capture. It costs a
  one-off Inductor compile pause of a minute or two on the first steady-state frame, and its
  fused kernels are not the reference's, so it is off by default.

## Checkpoint

`step_200000.pt` is the checkpoint every MyoCodec number in the paper uses.

| file | contents | resumable |
|---|---|---|
| `myocodec_step200000_model.pt` | generator weights only (12.72 M params) | no |
| `myocodec_step200000_full.pt` | generator + discriminator + both optimizer states | yes |

The weights are distributed with the submission's supplementary material rather than
committed to this repository. Place both files at the repository root, or pass absolute
paths, for the commands above to work as written.

```python
import torch
from streaming_emg_codec.config import load_config
from streaming_emg_codec.model import StreamingEMGCodec

cfg = load_config("configs/pretrain_stage2.yaml")
model = StreamingEMGCodec(cfg.model)
model.load_state_dict(torch.load("myocodec_step200000_model.pt", map_location="cpu")["model"])
model.eval()

# x: [batch, channels, samples] at 2 kHz, normalized per window across channels
out = model(x)                      # out["reconstruction"], out["indices"]
```

## Notes

- **Normalization is per-window and across-channel**: one mean and one standard deviation
  shared over channels and time. Per-channel normalization erases relative amplitude between
  channels, which the codec is meant to preserve.
- **Input rate is 2 kHz.** Resample anything else, anti-aliased, before encoding.
- **Windows are 5.0 s** (10,000 samples at 2 kHz) throughout training and evaluation.
- The optimizer is **Muon on 2-D hidden matrices, AdamW on everything else** — the RVQ
  codebooks are lookup tables, not linear maps, and the input/output projections touch the
  data boundary, so both stay on AdamW along with all 1-D parameters.
- **Encode in fp32 if you need a reproducible token stream.** Streaming and offline encoding
  agree on 99.98% of frames in fp32, but only 95.7% under the bfloat16 autocast used for
  training: RVQ assigns each frame by nearest neighbour, so a ~1e-4 difference flips frames
  sitting near a Voronoi boundary, and the residual structure compounds the flip down the
  codebooks. Reconstruction is barely affected. `tools/verify_streaming.py --amp` reports
  both.
- **Quote a specific `step_*.pt`, not "the final model".** Codebook occupancy drifts across
  saves, so RVQ numbers are not reproducible without pinning the checkpoint.

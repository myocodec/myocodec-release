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
streaming_emg_codec/
  config.py       typed config; unknown YAML keys are an error, not a silent default
  train.py        two-stage trainer (Muon/AdamW split, AMP, disc ramp, resume)
  losses.py       Huber + multi-scale spectral + VQ + commitment
  model/
    codec.py      encoder / RVQ / decoder, framing, and the streaming interface
    attention.py  sliding-window causal attention with a bounded KV cache
    rvq.py        residual VQ with dead-entry restart
    discriminator.py   multi-period + multi-scale-STFT discriminator
  data/
    readers.py       per-format readers, one per source corpus
    build_corpus.py  raw corpora -> unified 2 kHz HDF5 + manifest, with splits
    conditioning.py  per-corpus preprocessing table (PAPER_SPEC) and causal filters
    extract_shards.py  conditioned, normalized 5 s windows -> h5 shards
    shards.py        the training dataset (block-wise mmap streamer)
tools/
  eval_recon.py       reconstruction metrics; the script behind every number in Table 5
  verify_streaming.py streamed vs offline equivalence on a trained checkpoint
tests/
  test_smoke.py           end-to-end, no data and no GPU needed
  test_streaming_cache.py attention cache: equivalence, bounded memory, no lookahead
docs/DATA.md        corpus composition, splits, and per-corpus preprocessing
```

## Install

```bash
pip install -r requirements.txt
```

`flash-attn` is optional. Without it the attention layer falls back to `torch` SDPA, which
reproduces the same `window_size=(W, 0)` sliding-window causal mask — verified to
**1.8e-07 max absolute difference in fp32** against the reference implementation. Attention
runs over frames (100–250 tokens for a 5 s window), not raw samples, so the fallback costs
essentially nothing.

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
  `tools/verify_streaming.py --amp` reports both.

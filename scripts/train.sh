#!/usr/bin/env bash
# Launch the trainer with the environment settings the released runs used.
#
#   scripts/train.sh --config configs/pretrain_stage1.yaml --resume auto
#
# Everything after the script name is forwarded to streaming_emg_codec.train.
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Python block-buffers stdout when it is redirected to a file, which makes a step-based
# progress check read a log that lags hundreds of steps behind the real run.
export PYTHONUNBUFFERED=1

# The dataloader is I/O bound on large sequential reads; oversubscribing BLAS threads in
# the workers costs more than it buys.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

# Stage 2 allocates and frees the discriminator's activations every step; without this the
# caching allocator fragments over a multi-day run and eventually OOMs at a batch size that
# ran fine for days.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python -m streaming_emg_codec.train "$@"

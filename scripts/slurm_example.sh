#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 2-00:00:00
#SBATCH -J streemg
#SBATCH --signal=B:USR1@180
#
# Example self-resubmitting SLURM chain. Full pretraining is far longer than any single
# job's time limit, so the job re-submits itself on the USR1 signal the scheduler sends
# before the walltime expires, and the trainer resumes from `latest.pt`.
#
# STAGE SELECTION IS AUTOMATIC, and deliberately so: resuming a post-100k checkpoint with
# the stage-1 config instantiates the discriminator at stage 1's batch size of 1024 and
# OOMs immediately. Reading the stage from the checkpoint's own step makes that
# unrepresentable.
REPO="${REPO:-$PWD}"
CKDIR="${CKDIR:-$REPO/checkpoints/streemg}"

resub() { sbatch "$REPO/scripts/slurm_example.sh"; echo "resubmitted next chain job"; }
trap resub USR1

step=0
if [ -f "$CKDIR/latest.pt" ]; then
  step=$(python - "$CKDIR/latest.pt" <<'PY' 2>/dev/null
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu").get("step", 0)))
except Exception:
    print(0)
PY
)
  step=${step:-0}
fi

if [ "$step" -lt 100000 ]; then
  CFG="$REPO/configs/pretrain_stage1.yaml"
  echo "step=$step -> STAGE 1 (generator only, batch 1024, to 100k)"
else
  CFG="$REPO/configs/pretrain_stage2.yaml"
  echo "step=$step -> STAGE 2 (discriminator active, batch 192)"
fi
echo "config: $CFG"

bash "$REPO/scripts/train.sh" --config "$CFG" --resume auto \
  --override train.output_dir="$CKDIR" &
wait

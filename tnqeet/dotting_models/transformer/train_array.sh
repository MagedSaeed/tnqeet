#!/bin/bash

mkdir -p tnqeet/dotting_models/transformer/training_logs

LAYERS=(3 6 9 12)
LAST_IDX=$((${#LAYERS[@]} - 1))

sbatch --array=0-${LAST_IDX} \
       --cpus-per-task=24 \
       --mem=64G \
       --gres=gpu:1 \
       --partition=A100 \
       --output=tnqeet/dotting_models/transformer/training_logs/slurm_run_%A_%a.out \
       --error=tnqeet/dotting_models/transformer/training_logs/slurm_run_%A_%a.err \
       --wrap="set -- ${LAYERS[*]}; shift \$SLURM_ARRAY_TASK_ID; PYTHONUNBUFFERED=1 uv run python tnqeet/dotting_models/transformer/train.py --num_layers \$1"

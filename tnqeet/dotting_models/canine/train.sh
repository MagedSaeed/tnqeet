#!/bin/bash

mkdir -p tnqeet/dotting_models/canine/training_logs

sbatch --cpus-per-task=24 \
       --mem=64G \
       --gres=gpu:1 \
       --partition=A100 \
       --output=tnqeet/dotting_models/canine/training_logs/slurm_run_%j.out \
       --error=tnqeet/dotting_models/canine/training_logs/slurm_run_%j.err \
       --wrap="PYTHONUNBUFFERED=1 uv run python tnqeet/dotting_models/canine/train.py"

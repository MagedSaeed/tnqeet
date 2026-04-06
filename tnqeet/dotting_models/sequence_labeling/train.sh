#!/bin/bash

for num_layers in {1..5}
do
    sbatch --cpus-per-task=24 \
           --mem=32G \
           --gres=gpu:1 \
           --output=tnqeet/dotting_models/sequence_labeling/training_logs/slurm_run_${num_layers}.out \
           --error=tnqeet/dotting_models/sequence_labeling/training_logs/slurm_run_${num_layers}.err \
           --partition=A100 \
           --wrap="PYTHONUNBUFFERED=1 uv run python tnqeet/dotting_models/sequence_labeling/train.py --num_layers ${num_layers}"
done
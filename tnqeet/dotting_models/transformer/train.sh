#!/bin/bash

mkdir -p tnqeet/dotting_models/transformer/training_logs

for num_layers in 3 6 9 12
do
    sbatch --cpus-per-task=24 \
           --mem=64G \
           --gres=gpu:1 \
           --partition=A6000 \
           --output=tnqeet/dotting_models/transformer/training_logs/slurm_run_layers_${num_layers}_%j.out \
           --error=tnqeet/dotting_models/transformer/training_logs/slurm_run_layers_${num_layers}_%j.err \
           --wrap="PYTHONUNBUFFERED=1 uv run python tnqeet/dotting_models/transformer/train.py --num_layers ${num_layers}"
done

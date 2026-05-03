#!/bin/bash

MODEL_TYPE="canine-c"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODEL_NAME="google/${MODEL_TYPE}"

mkdir -p tnqeet/dotting_models/canine/training_logs

sbatch --cpus-per-task=24 \
       --mem=64G \
       --gres=gpu:1 \
       --partition=A100 \
       --output=tnqeet/dotting_models/canine/training_logs/slurm_run_%j.out \
       --error=tnqeet/dotting_models/canine/training_logs/slurm_run_%j.err \
       --wrap="PYTHONUNBUFFERED=1 uv run python tnqeet/dotting_models/canine/train.py --model_name ${MODEL_NAME}"

#!/bin/bash
# Stage 1: Train Teacher Model with full modalities
# Implements Section 4.1: Two-stage training strategy

# Set environment variables for model paths (optional, can also set in config)
export CLIP_MODEL_DIR="${CLIP_MODEL_DIR:-./models/clip-vit-base-patch16}"
export VIT_MODEL_DIR="${VIT_MODEL_DIR:-./models/vit-base-patch16-224}"
export DATA_DIR="${DATA_DIR:-./data}"

python src/main.py \
    --mode teacher \
    --config configs/teacher_config.yaml \
    --data-dir "${DATA_DIR}" \
    --checkpoint-dir checkpoints/teacher \
    --log-dir logs/teacher \
    --device cuda:0 \
    --epochs 30 \
    --batch-size 16 \
    --lambda-phys 0.05 \
    --lambda-proxy 0.0 \
    --lambda-ranking 0.1


#!/bin/bash
# Stage 2: Distill Student Model from Teacher
# Implements Section 4.1: Knowledge distillation for "Open Web" deployment

# Set environment variables for model paths (optional, can also set in config)
export CLIP_MODEL_DIR="${CLIP_MODEL_DIR:-./models/clip-vit-base-patch16}"
export VIT_MODEL_DIR="${VIT_MODEL_DIR:-./models/vit-base-patch16-224}"
export DATA_DIR="${DATA_DIR:-./data}"

python src/main.py \
    --mode student \
    --config configs/student_config.yaml \
    --data-dir "${DATA_DIR}" \
    --teacher-checkpoint checkpoints/teacher/best.pth \
    --checkpoint-dir checkpoints/student \
    --log-dir logs/student \
    --device cuda:0 \
    --epochs 30 \
    --batch-size 4 \
    --lambda-kd 0.1 \
    --lambda-fd 0.05 \
    --lambda-phys 0.2 \
    --lambda-proxy 0.3 \
    --lambda-ranking 0.1


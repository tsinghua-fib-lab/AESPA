# AESPA:  a physics- aware multimodal framework for tract-level urban LST estimation that combines satellite imagery, street-view panoramas, mobility- derived activity profiles, and interpretable physical proxies

AESPA is a deep learning framework for predicting Land Surface Temperature (LST) using multi-modal data (satellite imagery, street view images, and mobility patterns) with physics-aware constraints.

## Overview

AESPA employs a two-stage training approach:
1. **Teacher Model**: Trained with full modalities (satellite, street view, mobility)
2. **Student Model**: Distilled from teacher, using only visual modalities (satellite, street view)

Key features:
- **Cross-Attention Fusion**: Satellite features attend to street view features
- **FiLM Modulation**: Conditional feature modulation based on temporal/contextual tokens
- **Concept Bottleneck**: Weak supervision via 5 physical proxy indicators (NDVI, tree canopy, impervious surface, albedo, shadow)
- **Physics Consistency Loss**: Enforces physical relationships (e.g., NDVI ↑ → LST ↓)
- **Ranking Loss**: Ensures day temperature > night temperature
- **Knowledge Distillation**: Transfers knowledge from teacher to student

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Before training, you need to set up paths for pre-trained models and data. You have two options:

### Option 1: Set Environment Variables (Recommended)

```bash
# Model paths (download CLIP and ViT models first)
export CLIP_MODEL_DIR="/path/to/clip-vit-base-patch16"
export VIT_MODEL_DIR="/path/to/vit-base-patch16-224"

# Data directory
export DATA_DIR="/path/to/your/data"
```

Then update the config files to use these variables, or directly replace `${CLIP_MODEL_DIR}`, `${VIT_MODEL_DIR}`, and `${DATA_DIR}` in the YAML files.

### Option 2: Edit Config Files Directly

Edit `configs/teacher_config.yaml` and `configs/student_config.yaml` and replace:
- `${CLIP_MODEL_DIR:-./models/clip-vit-base-patch16}` with your CLIP model path
- `${VIT_MODEL_DIR:-./models/vit-base-patch16-224}` with your ViT model path  
- `${DATA_DIR:-./data}` with your data directory path

## Data Preparation

1. Organize your data according to `data/readme.md`
2. Preprocess physical proxy indicators:

```bash
# Set DATA_DIR environment variable or use relative path
export DATA_DIR="/path/to/your/data"

python data/preprocess_proxies.py --data-dir "${DATA_DIR:-./data}" --split train
python data/preprocess_proxies.py --data-dir "${DATA_DIR:-./data}" --split val
python data/preprocess_proxies.py --data-dir "${DATA_DIR:-./data}" --split test
```

## Training

### Stage 1: Train Teacher Model

```bash
bash scripts/train_teacher.sh
```

### Stage 2: Distill Student Model

```bash
bash scripts/distill_student.sh
```

## Model Architecture

- **Encoders**:
  - Satellite: ViT-Base with Adapter layers
  - Street View: CLIP ViT-B/16 with LoRA and Gated Attention MIL (Multiple Instance Learning)
  - Mobility: GRU network for 24×7 (168-dim) mobility patterns (teacher only)

- **Fusion**:
  - Image Cross Attention: Satellite features attend to street view features
  - FiLM (Feature-wise Linear Modulation): Injects Proxy and Mobility vectors into visual features
  - Adaptive gating for conditional tokens

- **Heads**:
  - Concept Bottleneck (proxy prediction)
  - LST Regression Head (temperature prediction)

## Loss Functions

- **Prediction Loss**: L1 loss for LST prediction
- **Physics Consistency Loss**: Enforces physical relationships
- **Ranking Loss**: Day > Night temperature constraint
- **Distillation Loss**: Feature + Knowledge distillation (student training)

## Citation

If you use AESPA in your research, please cite:

```bibtex
@article{aespa2024,
  title={AESPA: Adaptive Environmental Sensing with Physics-Aware Learning},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Your License Here]




# AESPA

A PyTorch implementation for the paper: **Physics-Aware Multimodal Urban Heat Mapping with Open Web Imagery and Mobility Data**.

**Anonymous Author(s)**

The repo currently includes code implementations for the following tasks:

  * **Multimodal LST Prediction:** Integrates satellite imagery, street-view panoramas, and mobility profiles for tract-level temperature estimation.
  * **Physics-Aware Modeling:** Incorporates physical proxies (vegetation, albedo, shadow, etc.) to enforce monotonic consistency and interpretability.
  * **Cross-City Generalization:** Utilizes a Teacher-Student distillation framework to transfer mobility knowledge to imagery-only models, enabling robust deployment in unseen cities.
  * **Socioeconomic Analysis:** Capable of revealing intra-urban heat disparities across neighborhoods with different socioeconomic characteristics.

## 🎉 Updates

  * **📢: News (2025.)** The code for **AESPA** is released.
  * **📢: News (2025.)** This paper has been submitted to **Web4Good 2026**.

## Introduction

🏆 Extreme urban heat is intensifying worldwide. **AESPA** (Aligned Environmental Sensing with Physics-aware Attribution) is a multimodal framework that combines satellite imagery, street-view panoramas, and mobility-derived activity profiles to estimate fine-grained Land Surface Temperature (LST).

By utilizing **Physics-Aware Regularization** and **Knowledge Distillation**, AESPA breaks the "black box" nature of deep learning models, ensuring physical plausibility while enabling deployment in data-poor cities where mobility data may be unavailable.

<img width="1010" height="484" alt="485f0da1-64d2-4479-86b8-16af21666aa4" src="https://github.com/user-attachments/assets/523c0fdd-3ab3-4d3f-b7b5-f4ab8bfd5842" />


## Overall Architecture

🌟 The training of AESPA consists of two stages: (i) **Mobility-Aware Teacher Training**, which fuses all modalities and learns physical proxies, and (ii) **Imagery-Only Student Distillation**, which learns to mimic the teacher's predictions and feature representations using only visual data.

The core components include:

  * **Encoders:** ViT (Satellite), CLIP+MIL (Street View), GRU (Mobility).
  * **Fusion:** Cross-feature fusion with FiLM-style conditioning.
  * **Physics Constraints:** Auxiliary heads and loss functions for vegetation, canopy, imperviousness, albedo, and shadow.

## ⚖ Performance Comparison

Comparison with baseline models across 8 major U.S. metropolitan areas (MSAs):

<img width="2438" height="746" alt="86e7c55d-4aac-492c-bd1b-13d865257c3e" src="https://github.com/user-attachments/assets/f832abf6-cb7a-43fc-8cea-59cf0aec4907" />


## Data

We use data from 8 U.S. MSAs (Dallas, Washington, Miami, Boston, Seattle, Minneapolis, St. Louis, Pittsburgh) to demonstrate AESPA.

  * **Satellite Imagery:** Web-based mapping platforms (Esri).
  * **Street View:** Google Street View API (panoramas).
  * **Mobility:** SafeGraph Weekly Patterns.
  * **Labels:** US Surface Urban Heat Island database (Summer Daytime LST).

Please refer to `data/readme.md` for preprocessing scripts and data structure details.

## ⚙️ Installation

### Environment

  * Tested OS: Linux
  * Python \>= 3.9
  * PyTorch \>= 2.0.0
  * CUDA (recommended for GPU training)

### Dependencies

1.  Install PyTorch with the correct CUDA version from the [PyTorch official website](https://pytorch.org/).

2.  Install all Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    The main dependencies include:

      * `torch>=2.0.0`
      * `transformers>=4.30.0` (for CLIP and ViT models)
      * `peft>=0.4.0` (for LoRA fine-tuning)
      * `tensorboard>=2.13.0` (for training visualization)

### Configuration

Before training, set up the following environment variables or update the config files:

```bash
# Model paths (download CLIP and ViT models first)
export CLIP_MODEL_DIR="/path/to/clip-vit-base-patch16"
export VIT_MODEL_DIR="/path/to/vit-base-patch16-224"

# Data directory
export DATA_DIR="/path/to/your/data"
```

Alternatively, you can directly edit the paths in `configs/teacher_config.yaml` and `configs/student_config.yaml`.

## 🏃 Model Training

Please first navigate to the root directory of the project.

### Data Preparation

1.  Organize your data according to `data/readme.md`.
2.  Preprocess physical proxy indicators (Vegetation, Albedo, etc.) from street view imagery:

<!-- end list -->

```bash
# Set DATA_DIR environment variable or use relative path
export DATA_DIR="/path/to/your/data"

python data/preprocess_proxies.py --data-dir "${DATA_DIR:-./data}" --split train
python data/preprocess_proxies.py --data-dir "${DATA_DIR:-./data}" --split val
python data/preprocess_proxies.py --data-dir "${DATA_DIR:-./data}" --split test
```

### Stage-1: Teacher Model Training (with Mobility)

We provide the training script `scripts/train_teacher.sh`. You can train the teacher model which uses **Satellite**, **Street View**, and **Mobility** data:

```bash
bash scripts/train_teacher.sh
```

Or run directly with Python:

```bash
python src/main.py \
    --mode teacher \
    --config configs/teacher_config.yaml \
    --data-dir /path/to/data \
    --checkpoint-dir checkpoints/teacher \
    --log-dir logs/teacher \
    --device cuda:0 \
    --epochs 30 \
    --batch-size 16 \
    --lambda-phys 0.05 \
    --lambda-proxy 0.0 \
    --lambda-ranking 0.1
```

**Parameters:**

  * `--mode`: Training mode (`teacher` or `student`).
  * `--config`: Path to YAML configuration file.
  * `--data-dir`: Path to data directory.
  * `--checkpoint-dir`: Directory to save model checkpoints.
  * `--log-dir`: Directory for TensorBoard logs.
  * `--lambda-phys`: Weight for physics consistency loss (default: 0.05).
  * `--lambda-ranking`: Weight for day-night ranking loss (default: 0.1).

Once trained, you will find logs in `logs/teacher/` and the best model in `checkpoints/teacher/best.pth`.

### Stage-2: Student Model Distillation (Imagery Only)

We provide the distillation script `scripts/distill_student.sh`. The student model learns from the frozen teacher and only uses **Satellite** and **Street View** inputs (no mobility data needed for inference):

```bash
bash scripts/distill_student.sh
```

Or run directly with Python:

```bash
python src/main.py \
    --mode student \
    --config configs/student_config.yaml \
    --data-dir /path/to/data \
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
```

**Additional Parameters for Student:**

  * `--teacher-checkpoint`: Path to the pre-trained teacher model checkpoint.
  * `--lambda-kd`: Weight for Knowledge Distillation loss (logits matching).
  * `--lambda-fd`: Weight for Feature Distillation loss (feature matching).
  * **Note:** Student training uses a smaller batch size (e.g., 4) because both teacher and student models are loaded into VRAM simultaneously.

## Model Weights

We provide downloads of model weights on [Link Coming Soon].

## 👀 Citation

If you find this repo helpful, please cite our paper:

```bibtex
@inproceedings{anonymous2026aespa,
  title={Physics-Aware Multimodal Urban Heat Mapping with Open Web Imagery and Mobility Data},
  author={Anonymous Author(s)},
  booktitle={Web4Good 2026 Submission},
  year={2026}
}
```

## 🙇‍ Acknowledgement

We appreciate the following resources:

  * **SafeGraph:** For providing human mobility data.
  * **US Surface Urban Heat Island Database:** For ground truth LST data.
  * **CLIP & ViT:** For the visual backbone implementations.

## 📧 Contact

If you have any questions or want to use the code, feel free to contact:

[yyy25@mails.tsinghua.edu.cn]

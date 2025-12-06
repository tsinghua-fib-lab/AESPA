
-----

# AESPA

A pytorch implementation for the paper: **Physics-Aware Multimodal Urban Heat Mapping with Open Web Imagery and Mobility Data**.

**Anonymous Author(s)**

The repo currently includes code implementations for the following tasks:

  * **Multimodal LST Prediction:** Integrates satellite imagery, street-view panoramas, and mobility profiles for tract-level temperature estimation.
  * **Physics-Aware Modeling:** Incorporates physical proxies (vegetation, albedo, shadow, etc.) to enforce monotonic consistency and interpretability.
  * **Cross-City Generalization:** Utilizes a Teacher-Student distillation framework to transfer mobility knowledge to imagery-only models, enabling robust deployment in unseen cities.
  * **Socioeconomic Analysis:** Capable of revealing intra-urban heat disparities across neighborhoods with different socioeconomic characteristics.

## 🎉 Updates

  * **📢: News (2025.xx)** The code for **AESPA** is released.
  * **📢: News (2025.xx)** This paper has been submitted to **Web4Good 2026**.

## Introduction

🏆 Extreme urban heat is intensifying worldwide. **AESPA** (Aligned Environmental Sensing with Physics-aware Attribution) is a multimodal framework that combines satellite imagery, street-view panoramas, and mobility-derived activity profiles to estimate fine-grained Land Surface Temperature (LST).

By utilizing **Physics-Aware Regularization** and **Knowledge Distillation**, AESPA breaks the "black box" nature of deep learning models, ensuring physical plausibility while enabling deployment in data-poor cities where mobility data may be unavailable.

## Overall Architecture

🌟 The training of AESPA consists of two stages: (i) **Mobility-Aware Teacher Training**, which fuses all modalities and learns physical proxies, and (ii) **Imagery-Only Student Distillation**, which learns to mimic the teacher's predictions and feature representations using only visual data.

The core components include:

  * **Encoders:** ViT (Satellite), CLIP+MIL (Street View), GRU (Mobility).
  * **Fusion:** Cross-feature fusion with FiLM-style conditioning.
  * **Physics Constraints:** Auxiliary heads and loss functions for vegetation, canopy, imperviousness, albedo, and shadow.

## ⚖ Performance Comparison

Comparison with baseline models across 8 major U.S. metropolitan areas (MSAs):

| Model | Modality | Physics-Aware | Mobility-Guided | MAE ($^\circ$C) | Pearson ($\rho$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet** | Sat + SV | ✗ | ✗ | 1.95 | 0.61 |
| **Tile2Vec** | Sat + SV | ✗ | ✗ | High | Low |
| **UrbanHeat** | Sat Only | ✗ | ✗ | High | Low |
| **Proxy+Reg** | SV Only | ✗ | ✗ | High | Low |
| **AESPA (Ours)** | **Sat + SV** | **✓** | **✓ (Distilled)** | **1.33** | **0.76** |

## Data

We use data from 8 U.S. MSAs (Dallas, Washington, Miami, Boston, Seattle, Minneapolis, St. Louis, Pittsburgh) to demonstrate AESPA.

  * **Satellite Imagery:** Web-based mapping platforms (Esri).
  * **Street View:** Google Street View API (panoramas).
  * **Mobility:** SafeGraph Weekly Patterns.
  * **Labels:** US Surface Urban Heat Island database (Summer Daytime LST).

Please refer to `data/readme.md` for preprocessing scripts.

## ⚙️ Installation

### Environment

  * Tested OS: Linux
  * Python \>= 3.9
  * torch \>= 2.0.0
  * Tensorboard

### Dependencies:

1.  Install Pytorch with the correct CUDA version.
2.  Use the `pip install -r requirements.txt` command to install all of the Python modules and packages used in this project.

## 🏃 Model Training

Please first navigate to the `src` directory: `cd src`
Then create a folder named `experiments` to record the training process: `mkdir experiments`

### Stage-1: Teacher Model Training (with Mobility)

We provide the scripts under the folder `./scripts/train_teacher.sh`. You can train the teacher model which uses Satellite, Street View, and Mobility data:

```bash
python main.py --device_id 0 --mode teacher \
  --dataset Dallas*Boston*Miami \
  --lambda_phys 0.05 --lambda_proxy 0.0 --lambda_rank 0.1 \
  --lr 1e-4 --weight_decay 0.05 \
  --batch_size 32
```

Once your model is trained, you will find the logs in `./logs/`. The trained teacher model will be saved in `./experiments/Teacher_<dataset>/model_save/model_best.pkl`.

### Stage-2: Student Model Distillation (Imagery Only)

We provide the scripts under the folder `./scripts/distill_student.sh`. The student model learns from the frozen teacher and only uses Satellite and Street View inputs:

```bash
python main.py --device_id 0 --mode student \
  --dataset Dallas*Boston*Miami \
  --teacher_path ./experiments/Teacher_<dataset>/model_save/model_best.pkl \
  --lambda_phys 0.2 --lambda_proxy 0.3 --lambda_rank 0.1 \
  --lambda_kd 0.1 --lambda_fd 0.05 \
  --lr 1e-4
```

**Parameters to specify:**

  * `teacher_path`: Path to the pre-trained teacher model.
  * `lambda_phys`: Weight for physics consistency loss (sign-constrained correlation).
  * `lambda_proxy`: Weight for proxy reconstruction loss.
  * `lambda_rank`: Weight for day-night ranking loss.
  * `lambda_kd` & `lambda_fd`: Weights for Knowledge Distillation (prediction) and Feature Distillation.

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

[Your Name/Email Placeholder]

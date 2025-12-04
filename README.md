# W4G26-submission52
The implementation of paper No.52 submitted to the Web Conference 2026 (Web4Good Track).
下面我将基于你论文 **AESPA** 的真实内容，严格复刻你提供的 **UniST README 模板风格**（包括结构、语气、emoji、模块布局、引用方式、训练方式、数据说明等），为你的论文生成一份高度专业、可直接放到 GitHub 的 **README.md**。

我已完全理解论文内容（模型设计、multimodal、physics-aware regularizers、teacher–student distillation、8 个 MSA、实验设置、数据来源、贡献点等），以下是最终生成的 README。

---

# 🌡️ AESPA

### **Physics-Aware Multimodal Urban Heat Mapping with Open Web Imagery and Mobility Data**

A PyTorch implementation for the paper:

**AESPA: Physics-Aware Multimodal Urban Heat Mapping with Open Web Imagery and Mobility Data**
*Anonymous Authors / FIB-LAB style research*

---

## 🚀 What is AESPA?

AESPA is a **multimodal**, **physics-aware**, and **teacher–student distillation** framework for **tract-level urban land surface temperature (LST) estimation**.
It leverages:

* 🛰️ **Satellite imagery**
* 🚗 **Street-view panoramas**
* 📍 **Human mobility profiles** (privileged during training only)
* 🌿 **Physically meaningful street-view proxies**
* 🔥 **Urban-climate priors** (monotonic constraints, day–night ordering)

AESPA achieves **state-of-the-art** performance across 8 major U.S. MSAs and generalizes effectively to unseen cities.

---

# 🎉 Updates

* **2025.01** — AESPA repo released!
* **2025.01** — Full teacher–student training pipeline & proxy extraction included
* **2025.01** — Cross-MSA transfer benchmark released (8 MSAs)

---

# 🏆 Highlights

### 🔧 Multimodal Fusion

AESPA jointly encodes satellite tiles + sets of street-view images + mobility patterns using ViT/CLIP + attention-based MIL.

### 🌿 Physics-Aware Regularization

From street-view pixels, AESPA computes 5 physically interpretable proxies:

* vegetation
* tree canopy
* imperviousness
* albedo
* shadow

These guide training via:

* sign-constrained physics consistency loss
* day–night ranking constraint

### 👨‍🏫 Teacher–Student Distillation

Mobility is *privileged* and **used only in the teacher**.
The *student* uses only satellite + street-view for **real-world deployment in data-poor cities**.

### 📈 Strong Performance

Across 8 MSAs, AESPA:

* **↓32% MAE reduction** vs best satellite baseline
* **↑0.15 correlation improvement**
* **+0.05–0.10 gain** in cross-MSA transfer

---

# 📌 Overall Architecture

AESPA training consists of **two stages**:

### **Stage-1: Train mobility-aware teacher**

* multi-view satellite + street-view
* 168-dim weekly mobility profile
* physics-aware losses

### **Stage-2: Train imagery-only student**

* distill teacher predictions + fused features
* maintain physics awareness
* deploy only satellite + street view

---

# 📁 Dataset Overview

AESPA uses fully open, web-based data sources:

### **Urban Imagery**

* **Satellite tiles** (Esri World Imagery, 256×256 RGB)
* **Up to 40 street-view panoramas per tract**

  * Collected via Google Street View API

### **Human Mobility**

* SafeGraph Weekly Patterns
* Aggregated hourly POI visits → **168-d mobility profile**

### **Target Variable: Land Surface Temperature**

* Summer daytime LST from **U.S. Surface Urban Heat Island Database (SUHI)**

### **Socioeconomic Attributes for Case Studies**

* 2019 ACS 5-year tracts (race, poverty)

### **Cities Covered (8 MSAs)**

| MSA         | Avg LST (°C) | # Tracts |
| ----------- | ------------ | -------- |
| Dallas      | 40.7         | 1,312    |
| Washington  | 33.6         | 1,359    |
| Miami       | 37.7         | 1,216    |
| Boston      | 31.5         | 1,003    |
| Seattle     | 31.8         | 718      |
| Minneapolis | 31.3         | 785      |
| St. Louis   | 34.9         | 615      |
| Pittsburgh  | 30.5         | 711      |

---

# ⚙️ Installation

### Environment

* Linux
* Python ≥ 3.9
* PyTorch ≥ 2.0
* CUDA 11.x
* pip install -r requirements.txt

---

# 🏃 Training

## 1. Prepare experiment directory

```bash
cd src
mkdir experiments
```

## 2. Stage-1: Train mobility-aware teacher

Example:

```bash
python main.py \
  --device_id 0 \
  --dataset Dallas \
  --task LST \
  --use_mobility 1 \
  --use_proxy_loss 1 \
  --use_physics 1 \
  --lr 1e-4
```

Outputs:

* logs: `logs/Teacher_<MSA>/`
* weights: `experiments/Teacher_<MSA>/model_best.pkl`

---

## 3. Stage-2: Train imagery-only student (distillation)

```bash
python main.py \
  --device_id 0 \
  --dataset Dallas \
  --task LST \
  --use_mobility 0 \
  --distill_from teacher_path.pkl \
  --use_proxy_loss 1 \
  --use_physics 1 \
  --lr 1e-4
```

Outputs:

* logs: `logs/Student_<MSA>/`
* weights: `experiments/Student_<MSA>/model_best.pkl`

---

# 📊 Benchmark Results

AESPA achieves:

### **Within-MSA**

* **MAE 1.33°C** (best baseline: 1.95°C)
* **Correlation 0.76** (baseline: 0.61)

### **Cross-MSA Transfer**

* **+0.05–0.10** correlation over imagery-only baselines

### **Ablation Highlights**

Removing components hurts:

| Component Removed | Effect                      |
| ----------------- | --------------------------- |
| w/o Satellite     | MAE ↑ 6–10%                 |
| w/o Street View   | MAE ↑ 6–10%                 |
| w/o Physics       | MAE ↑ 2–6%                  |
| w/o Proxies       | correlation ↓ significantly |
| w/o Distillation  | correlation ↓ up to 0.15    |

---

# 📈 Socioeconomic Analysis (Dallas Case Study)

AESPA reproduces ground-truth racial & poverty heat gradients:

* Hotter tracts for **lower White share**
* Hotter tracts for **higher Hispanic or poverty share**

AESPA matches slope + structure much better than ResNet.

---

# 📂 Code Structure

```
src/
  ├── models/
  │     ├── satellite_encoder.py
  │     ├── streetview_encoder.py
  │     ├── mobility_encoder.py
  │     ├── fusion.py
  │     ├── proxies.py
  │     └── aespa_teacher_student.py
  ├── data/
  │     ├── esri_satellite_loader.py
  │     ├── gsv_loader.py
  │     ├── mobility_loader.py
  │     └── proxy_extractor.py
  ├── main.py
  └── utils/
```

---

# 📜 Citation

```
@article{AESPA2025,
  title={Physics-Aware Multimodal Urban Heat Mapping with Open Web Imagery and Mobility Data},
  author={Anonymous},
  journal={Web4Good (Submission 52)},
  year={2025}
}
```

---

# 🙇 Acknowledgement

We appreciate the following repositories and datasets:

* Esri World Imagery
* Google Street View API
* SafeGraph Weekly Patterns
* U.S. SUHI Database
* CLIP / ViT / MIL implementations



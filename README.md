# 🧠 Brain Tumor MRI Classification

<p align="center">
  <img src="https://github.com/user-attachments/assets/6bbca955-31f5-4bc4-a498-7200270f5325" width="140" alt="Brain Tumor MRI Classification Logo"/>
</p>

<p align="center">
  <strong>End-to-End Brain MRI Classification & Deployment Pipeline</strong>
</p>

<p align="center">
  A production-oriented PyTorch project for classifying brain MRI images into four categories using EfficientNet-B3 transfer learning, MLflow experiment tracking, validated model export, and a FastAPI + ONNX web inference service.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch\&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow\&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?logo=docker\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</p>

---

## Overview

**Brain Tumor MRI Classification** is an end-to-end deep learning system designed to classify brain MRI images into four categories:

| Class        | Description      |
| ------------ | ---------------- |
| `glioma`     | Glioma tumor     |
| `meningioma` | Meningioma tumor |
| `notumor`    | No visible tumor |
| `pituitary`  | Pituitary tumor  |

The project focuses not only on model training, but on the complete machine-learning lifecycle:

```text
Dataset
   ↓
Data Validation & Preparation
   ↓
Stratified Train / Validation Split
   ↓
Data Augmentation
   ↓
EfficientNet-B3 Transfer Learning
   ↓
Weighted Cross-Entropy Training
   ↓
Validation & Early Stopping
   ↓
Best Model Checkpoint
   ↓
MLflow Experiment Tracking
   ↓
Independent Test Evaluation
   ↓
Model Export & Validation
   ↓
FastAPI + ONNX Inference
   ↓
Web Application / Cloud Deployment
```

---

## ✨ Key Features

* 🧠 **EfficientNet-B3** with ImageNet transfer learning
* 📊 Stratified train/validation splitting
* ⚖️ Class-weighted `CrossEntropyLoss`
* 🚀 Adam optimizer
* 📉 `ReduceLROnPlateau` learning-rate scheduling
* ⏹️ Early stopping with best-checkpoint restoration
* 🔬 MLflow experiment tracking
* 📈 Comprehensive test evaluation
* 📦 Multiple deployment-oriented model exports
* 🔄 ONNX Runtime inference validation
* 🖼️ Single-image prediction CLI
* 🌐 FastAPI inference backend
* 💻 Browser-based web interface
* 🐳 Docker support
* ☁️ AWS ECS / Fargate deployment support
* 🔐 Optional private S3 model loading
* 🧪 Automated test suite
* ⚙️ Centralized YAML configuration
* 🧰 `uv`-based dependency management

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │   Kaggle / Local    │
                         │    MRI Dataset      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Pipeline     │
                         │                     │
                         │ • Validation        │
                         │ • Stratified Split  │
                         │ • Augmentation      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   EfficientNet-B3   │
                         │ ImageNet Transfer   │
                         │      Learning       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Trainer        │
                         │                     │
                         │ • Weighted CE       │
                         │ • Adam              │
                         │ • LR Scheduler      │
                         │ • AMP               │
                         │ • Early Stopping    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │       Best Checkpoint        │
                    │       + MLflow Run            │
                    └──────────────┬───────────────┘
                                   │
                 ┌─────────────────┼──────────────────┐
                 │                 │                  │
                 ▼                 ▼                  ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │   Evaluation   │ │     Export     │ │   Inference    │
        │                │ │                │ │                │
        │ Accuracy       │ │ TorchScript    │ │ Single MRI     │
        │ Precision      │ │ PyTorch Lite   │ │ Prediction     │
        │ Recall         │ │ ONNX           │ │                │
        │ F1             │ │                │ │                │
        │ ROC-AUC        │ │                │ │                │
        └────────────────┘ └───────┬────────┘ └───────┬────────┘
                                   │                  │
                                   ▼                  ▼
                             ┌─────────────┐   ┌──────────────┐
                             │ ONNX Runtime│   │   FastAPI    │
                             │ Validation   │   │ Web Service  │
                             └─────────────┘   └──────┬───────┘
                                                       │
                                                       ▼
                                                ┌──────────────┐
                                                │ Web Browser   │
                                                └──────────────┘
```

---

# 📂 Project Structure

```text
.
├── configs/
│   └── config.yaml
│
├── data/
│   └── raw/
│       ├── Training/
│       └── Testing/
│
├── artifacts/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── history.json
│   │
│   ├── evaluation/
│   │   └── test_metrics.json
│   │
│   └── exports/
│       ├── brain_tumor_efficientnet_b3_web.pt
│       ├── brain_tumor_efficientnet_b3_gpu.pt
│       ├── brain_tumor_efficientnet_b3_mobile.ptl
│       ├── brain_tumor_efficientnet_b3.onnx
│       └── brain_tumor_efficientnet_b3_manifest.json
│
├── src/
│   └── brain_tumor/
│
├── tests/
│
├── scripts/
│   ├── setup.bash
│   └── run.bash
│
├── webapp/
│   ├── frontend/
│   └── backend/
│
├── aws/
│   └── README.md
│
├── start_webapp.bash
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

---

# 🚀 Quick Start

This project uses **[uv](https://docs.astral.sh/uv/)** for environment and dependency management.

### 1. Clone

```bash
git clone https://github.com/Amirmoh11n/ALL_IN_ONE_Project-V1/
cd ALL_IN_ONE_Project-V1
```

### 2. Setup

```bash
chmod +x scripts/setup.bash scripts/run.bash
./scripts/setup.bash
```

The setup script:

* Creates the virtual environment
* Installs project dependencies
* Creates required artifact directories
* Prepares the project for training

---

# 📦 Dataset

The pipeline supports automatic dataset acquisition when the expected dataset directories are missing.

Expected structure:

```text
data/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

If the dataset already exists under `data/raw/`, Kaggle authentication is not required.

### Kaggle Authentication

Using `kaggle.json`:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Or environment variables:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

---

# 🧪 Testing

Run the complete test suite:

```bash
uv run pytest tests -v
```

The tests cover core project functionality and help prevent regressions across the training, evaluation, export, and inference pipeline.

---

# 🎯 Training

Run the complete training pipeline:

```bash
uv run brain-tumor train
```

For a quick smoke test:

```bash
uv run brain-tumor train --epochs 2
```

Training pipeline:

```text
Dataset
   ↓
Stratified Split
   ↓
Augmentation
   ↓
DataLoader
   ↓
EfficientNet-B3
   ↓
Weighted CrossEntropy
   ↓
Adam
   ↓
ReduceLROnPlateau
   ↓
Validation
   ↓
Early Stopping
   ↓
Best Checkpoint
```

Outputs:

```text
artifacts/checkpoints/
├── best_model.pt
└── history.json
```

---

# 📊 MLflow Experiment Tracking

MLflow is used to track experiments, parameters, metrics, and model-related artifacts.

Start the local MLflow server:

```bash
./scripts/run.bash mlflow
```

Then open the URL printed by MLflow.

Experiment configuration is available in:

```text
configs/config.yaml
```

---

# 🔬 Final Evaluation

The final evaluation is performed against the **untouched Testing split**:

```bash
uv run brain-tumor evaluate
```

Results:

```text
artifacts/evaluation/test_metrics.json
```

### Metrics

The evaluation report includes:

* Accuracy
* Macro Precision
* Macro Recall / Sensitivity
* Macro F1
* Macro ROC-AUC (OvR)
* Per-class Precision
* Per-class Recall
* Per-class F1
* Confusion Matrix

### Results

> Add the final test results here after the final training run.

Example:

```text
Accuracy:        XX.XX%
Macro Precision: XX.XX%
Macro Recall:    XX.XX%
Macro F1:        XX.XX%
ROC-AUC (OvR):   XX.XX%
```

A confusion matrix can also be added here:

```text
                Predicted
              G   M   N   P
Actual   G    ·   ·   ·   ·
         M    ·   ·   ·   ·
         N    ·   ·   ·   ·
         P    ·   ·   ·   ·
```

---

# 📦 Model Export

After successful training:

```bash
uv run brain-tumor export
```

The exporter generates:

| Artifact          | Purpose                          |
| ----------------- | -------------------------------- |
| `*_web.pt`        | TorchScript inference            |
| `*_gpu.pt`        | GPU-oriented TorchScript         |
| `*_mobile.ptl`    | PyTorch Lite / mobile deployment |
| `*.onnx`          | Cross-platform inference         |
| `*_manifest.json` | Export metadata                  |

### ONNX Validation

The ONNX artifact is automatically validated using:

* `onnx.checker`
* ONNX Runtime inference
* PyTorch vs ONNX output comparison
* Dynamic batch-size testing

This helps ensure that the exported model is not only generated successfully, but is also executable and numerically consistent with the original model.

---

# 🖼️ Single-Image Inference

Run inference on a single MRI:

```bash
uv run brain-tumor predict \
  --image /absolute/path/to/mri.jpg
```

Example:

```json
{
  "predicted_class": "glioma",
  "confidence": 0.973,
  "probabilities": {
    "glioma": 0.973,
    "meningioma": 0.012,
    "notumor": 0.004,
    "pituitary": 0.011
  }
}
```

---

# 🌐 Web Application

The project includes a deployment-oriented web inference application.

### Stack

```text
Browser
   │
   ▼
HTML + Tailwind CSS + JavaScript
   │
   ▼
FastAPI
   │
   ▼
ONNX Runtime
   │
   ▼
EfficientNet-B3 ONNX Model
```

### Features

* MRI image upload
* PNG / JPG / JPEG / WEBP / BMP support
* 10 MB default upload limit
* Server-side model execution
* ONNX Runtime inference
* Health endpoint
* Readiness endpoint
* Docker support
* Cloud deployment support
* Optional private S3 model loading

The ONNX model is **never sent to the browser**.

### Start locally

After training and exporting:

```bash
./start_webapp.bash
```

The application will be available at:

```text
http://127.0.0.1:8000
```

---

# ☁️ Cloud Deployment

The web inference service is designed to support containerized deployment on AWS.

Supported deployment architecture:

```text
                    ┌───────────────┐
                    │    Browser    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     AWS       │
                    │ Load Balancer │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ ECS / Fargate │
                    │   FastAPI     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ ONNX Runtime  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Model / S3    │
                    └───────────────┘
```

For AWS deployment instructions:

```text
aws/README.md
```

The application can optionally load the model from a private S3 location using:

```text
MODEL_S3_URI
```

---

# ⚙️ Configuration

Experiment and runtime settings are centralized in:

```text
configs/config.yaml
```

Main sections:

```yaml
data:
model:
training:
evaluation:
export:
artifacts:
tracking:
logging:
```

Training hyperparameters are intentionally kept outside the source code to improve:

* Reproducibility
* Experiment management
* Configuration portability
* MLflow integration

---

# 🛠️ CLI

The main CLI commands are:

```bash
# Train
uv run brain-tumor train

# Evaluate
uv run brain-tumor evaluate

# Export
uv run brain-tumor export

# Predict
uv run brain-tumor predict --image /path/to/image.jpg

# Tests
uv run pytest tests -v
```

Or use the project runner:

```bash
./scripts/run.bash train
./scripts/run.bash evaluate
./scripts/run.bash export
./scripts/run.bash predict --image /path/to/image.jpg
./scripts/run.bash test
./scripts/run.bash mlflow
```

---

# 🔁 Reproducible ML Lifecycle

```text
┌──────────────────────────────────────────┐
│              DATA INGESTION              │
│          Kaggle / Local Dataset          │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│           DATA PREPARATION               │
│      Validation + Stratified Split       │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│              TRAINING                   │
│ EfficientNet-B3 + Transfer Learning     │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│         EXPERIMENT TRACKING              │
│                MLflow                    │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│              EVALUATION                  │
│      Untouched Testing Dataset           │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│               EXPORT                     │
│       TorchScript / PTL / ONNX           │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│              DEPLOYMENT                  │
│        FastAPI + ONNX Runtime            │
└──────────────────────────────────────────┘
```

---

# ⚠️ Medical & Dataset Limitations

This project is intended for **machine-learning engineering, experimentation, and research purposes**.

It is **not a medical diagnostic system** and model predictions must not be interpreted as a clinical diagnosis or used as a substitute for qualified medical professionals.

The final test set is kept separate from training and validation. However, the available dataset does not provide patient-level identifiers; therefore, the validation strategy is based on **image-level stratification rather than patient-level splitting**.

This limitation should be considered when interpreting model performance and generalization.

---

# 🔐 Security & Deployment Notes

For production deployment:

* Do not commit Kaggle credentials.
* Do not commit private cloud credentials.
* Keep production models in controlled storage.
* Prefer private S3 buckets for cloud-hosted models.
* Configure upload-size limits appropriately.
* Expose only required API endpoints.
* Use HTTPS in production.
* Keep dependencies updated.
* Do not expose the ONNX model directly to clients.

---

# 📌 Roadmap

Potential future improvements:

* [ ] Add Grad-CAM / explainability
* [ ] Add calibration analysis
* [ ] Add automated model versioning
* [ ] Add CI/CD pipeline
* [ ] Add Docker Compose development environment
* [ ] Add API authentication
* [ ] Add production monitoring
* [ ] Add inference latency benchmarking
* [ ] Add model performance dashboard
* [ ] Add patient-level evaluation when appropriate metadata becomes available

---

# 📜 License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

# 👨‍💻 Author

**Amirmohammad Nashalji**

Computer Engineering Student | Machine Learning & AI Enthusiast

---

<p align="center">
  Built with PyTorch, MLflow, ONNX Runtime, FastAPI, Docker and ❤️
</p>

# Brain Tumor MRI Classification
<img width="1254" height="1254" alt="logo2" src="https://github.com/user-attachments/assets/6bbca955-31f5-4bc4-a498-7200270f5325" />

End-to-end PyTorch project for classifying brain MRI images into four classes:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

The project uses **EfficientNet-B3 + ImageNet transfer learning**, stratified train/validation splitting, class-weighted CrossEntropyLoss, Adam, ReduceLROnPlateau, early stopping, MLflow tracking, full test evaluation, single-image inference, and validated model exports.

## 1. Setup

This project is designed for **uv**.

```bash
git clone <YOUR_REPOSITORY_URL>
cd ALL_IN_ONE_Project-V1

chmod +x scripts/setup.bash scripts/run.bash
./scripts/setup.bash
```

The setup script creates the virtual environment, installs all dependencies, and creates the artifact directories.

If you do not have `uv`, install it using the official installer/documentation, then rerun the setup script.

### Kaggle authentication

The pipeline automatically downloads the dataset when `data/raw/Training` and `data/raw/Testing` are missing.

Configure either:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

or set:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

If the dataset is already present under `data/raw/`, no Kaggle credentials are needed.

## 2. Run tests

```bash
uv run pytest tests -v
```

## 3. Train

The complete pipeline is:

`dataset -> stratified split -> augmentation -> DataLoader -> EfficientNet-B3 -> weighted CE -> Adam -> scheduler -> validation -> early stopping -> best checkpoint -> MLflow`

Run:

```bash
uv run brain-tumor train
```

For a quick experiment:

```bash
uv run brain-tumor train --epochs 2
```

The best checkpoint is saved to:

```text
artifacts/checkpoints/best_model.pt
```

Training history is saved to:

```text
artifacts/checkpoints/history.json
```

## 4. MLflow

Start the local MLflow UI:

```bash
./scripts/run.bash mlflow
```

Then open the address printed by MLflow in your browser.

The experiment is configured in `configs/config.yaml`.

## 5. Final evaluation

Evaluation uses the untouched `Testing` directory.

```bash
uv run brain-tumor evaluate
```

Output:

```text
artifacts/evaluation/test_metrics.json
```

The report contains:

- Accuracy
- Macro Precision
- Macro Recall / Sensitivity
- Macro F1
- Macro ROC-AUC OvR
- Per-class Precision
- Per-class Recall
- Per-class F1
- Confusion Matrix

## 6. Export

After a successful training run:

```bash
uv run brain-tumor export
```

The exporter produces and validates:

```text
artifacts/exports/
├── brain_tumor_efficientnet_b3_web.pt
├── brain_tumor_efficientnet_b3_gpu.pt
├── brain_tumor_efficientnet_b3_mobile.ptl
├── brain_tumor_efficientnet_b3.onnx
└── brain_tumor_efficientnet_b3_manifest.json
```

### Web/Cloud

TorchScript, loadable without the original Python model class.

### GPU

Frozen TorchScript for deployment-oriented inference.

### Mobile

PyTorch Lite Interpreter artifact. The exporter attempts mobile optimization and falls back to a plain traced module if the installed PyTorch build does not provide the required mobile optimization support.

### ONNX

ONNX is structurally checked with `onnx.checker`, executed with ONNX Runtime, compared against the PyTorch output, and tested with a dynamic batch size.

## 7. Single-image inference

```bash
uv run brain-tumor predict --image /absolute/path/to/mri.jpg
```

Example response:

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

## 8. Configuration

All experiment settings live in:

```text
configs/config.yaml
```

Important sections:

- `data`
- `model`
- `training`
- `evaluation`
- `export`
- `artifacts`
- `tracking.mlflow`
- `logging`

The source code does not hardcode training hyperparameters.

## 9. Project lifecycle

```text
                    ┌─────────────────┐
                    │ Kaggle / Local  │
                    │ Brain MRI Data  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Data Pipeline   │
                    │ train/val/test  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ EfficientNet-B3 │
                    │ Transfer Learn. │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Trainer         │
                    │ CE + Adam       │
                    │ Scheduler + AMP │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │ best_model.pt + MLflow run │
              └──────────────┬──────────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
        Evaluation        Export           Inference
        test_metrics      PT/PTL/ONNX      MRI -> JSON
```

## 10. Important medical-data note

This is a machine-learning engineering/research project. Model predictions must not be treated as a medical diagnosis. The untouched test split is used for final evaluation, but the dataset itself does not provide patient identifiers, so the validation strategy is image-level stratification rather than patient-level splitting.

## 11. Main commands

```bash
./scripts/run.bash train
./scripts/run.bash evaluate
./scripts/run.bash export
./scripts/run.bash predict --image /path/to/image.jpg
./scripts/run.bash test
./scripts/run.bash mlflow
```

## Web Application

The project includes a production-oriented web inference application:

- Frontend: HTML + Tailwind CSS + JavaScript
- Backend: FastAPI + ONNX Runtime
- Model execution: server-side only; the browser never receives the ONNX model
- Upload: PNG/JPG/JPEG/WEBP/BMP, max 10 MB by default
- Health/readiness endpoints for cloud load balancers
- Docker support
- AWS ECS Express Mode / Fargate deployment support
- Optional private S3 model loading through `MODEL_S3_URI`

After training and export:

```bash
./start_webapp.bash
```

The launcher starts FastAPI and opens the browser at `http://127.0.0.1:8000`.

For cloud deployment, see `aws/README.md`.

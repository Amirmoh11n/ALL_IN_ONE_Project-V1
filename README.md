# ALL_IN_ONE_Project-V1

End-to-end system for classifying brain radiology (MRI) images into 4 tumor classes, using the
Brain Tumor MRI Dataset (Masoud Nickparvar).

Delivers, incrementally (MVP first):
- A trained EfficientNet-B3 (PyTorch) model.
- A web application (FastAPI backend + React frontend) for uploading an MRI image and receiving a
  text prediction.

Every folder has its own short `README.md` describing its purpose in more detail — see:
`artifacts/`, `configs/`, `data/`, `docker/`, `scripts/`, `src/` (and its subpackages),
`tests/`, `webapplication/` (and its subfolders).

## Status

- ✅ **Step 1 — Data pipeline**: implemented and tested (`src/data/`). Downloads the
  dataset via `kagglehub` if missing, performs a stratified train/val split (15% val),
  builds `BrainTumorDataset`s with ImageNet normalization + on-the-fly augmentation
  (train only), and exposes ready-to-use `DataLoader`s through `DataPipeline`.
- ✅ **Step 2 — Model**: implemented and tested (`src/models/efficientnet.py`).
  `EfficientNetB3Classifier` wraps torchvision's ImageNet-pretrained EfficientNet-B3
  with a replaced 4-class output layer; built-in dropout preserved; `freeze_backbone`
  is config-driven (default: full fine-tune).
- ✅ **Step 3 — Training engine**: implemented and tested (`src/engine/trainer.py`).
  `Trainer` runs the full loop (CrossEntropyLoss, optionally class-weighted via
  runtime-computed weights, Adam, `ReduceLROnPlateau`, early stopping, best-checkpoint
  saving, optional MLflow tracking).
- ✅ **Step 4 — Metrics**: implemented and tested (`src/metrics/`). All 6 metrics
  (Confusion Matrix, Recall, F1-macro, Precision, ROC-AUC macro/OvR, Accuracy) as
  thin scikit-learn-backed classes with a consistent `.compute()` interface.
- ✅ **Step 5 — Evaluation**: implemented and tested (`src/evaluate/evaluate.py`).
  `ModelEvaluator` runs inference over a DataLoader (e.g. the untouched Testing
  set) and computes the full metric suite via `EvaluationResult`. Checkpoint
  loading (`src/utils/checkpoint.py`) is shared with the upcoming inference step.
- ⏳ Export, inference (endpoint wiring), and the web app are not yet
  implemented — pending step-by-step discussion and approval.

Run the data-pipeline tests:
```bash
pip install -e .
pytest tests/ -v
```

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
- ⏳ Model, training loop, evaluation, export, inference, and the web app are not
  yet implemented — pending step-by-step discussion and approval.

Run the data-pipeline tests:
```bash
pip install -e .
pytest tests/ -v
```

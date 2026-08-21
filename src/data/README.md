# src/data

Everything related to preparing data for the model:

- `dataset.py` — PyTorch `Dataset` class(es) for the MRI images.
- `classes.py` — the 4 tumor class labels and label/index mapping.
- `augment.py` — data augmentation pipeline (renamed from `agment.py`).
- `splitter.py` — Training/Validation split logic (renamed from `splliter.py`); implements the stratified
  image-level split (patient-level split is not feasible for this dataset — no patient IDs are provided).

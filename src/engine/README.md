# src/training

**New folder** (not present in the originally proposed tree; added because no component owned the actual
training loop).

- `trainer.py` — orchestrates the training loop: CrossEntropyLoss, optimizer, early stopping, LR scheduling,
  optional class weighting, and MLflow experiment tracking.

# tests

Unit/integration tests, one focused module per source module, using synthetic
(tiny, generated) images so tests run fast and don't need the real dataset or
network access.

- `conftest.py` — shared fixtures: builds fake `Training/<class>/*.jpg` and
  `Testing/<class>/*.jpg` folders with a few tiny generated images per class.
- `test_classes.py` — `TumorClasses` name/index mapping.
- `test_config_loader.py` — `ConfigLoader` dotted-path access.
- `test_downloader.py` — `DatasetDownloader` presence check + skip-download path
  (the actual kagglehub network download is out of scope for unit tests).
- `test_splitter.py` — `DatasetSplitter` ratio, stratification, disjointness,
  shuffling, and reproducibility.
- `test_dataset.py` — `BrainTumorDataset` loading, transforms, shapes.
- `test_pipeline.py` — end-to-end `DataPipeline` test (download-skip → split →
  dataset → dataloader) against a synthetic dataset.
- `test_efficientnet.py` — `EfficientNetB3Classifier` output shape, preserved
  dropout, full-fine-tune vs freeze_backbone behavior (uses `pretrained=False`
  to avoid a network dependency on ImageNet weights).
- `test_trainer.py` — `EarlyStopping`, `compute_class_weights`, and an end-to-end
  `Trainer` run (tiny dummy CNN + synthetic data) covering checkpoint saving,
  class-weighted loss, and MLflow metric logging.
- `test_metrics.py` — all 6 metrics against a hand-computed 4-class scenario
  (confusion matrix, recall, precision, F1 all checked against manually derived
  expected values, not just re-deriving sklearn's own answer), plus edge cases
  (perfect predictions, a class with zero predictions).
- `test_checkpoint.py` — `load_model_checkpoint` restores exact weights and
  sets the model to eval mode.
- `test_evaluate.py` — `ModelEvaluator` orchestration: correct confusion-matrix
  shape, macro metrics in valid range, per-class arrays sized correctly,
  `to_dict()` is JSON-serializable (exact metric *values* are covered by
  `test_metrics.py`, not repeated here).

Run with:
```bash
pytest tests/ -v
```

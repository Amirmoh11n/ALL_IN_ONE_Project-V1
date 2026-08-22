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

Run with:
```bash
pytest tests/ -v
```

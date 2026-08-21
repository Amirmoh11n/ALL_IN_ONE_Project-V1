# data

Local storage for the raw Brain Tumor MRI Dataset (Nickparvar). Not part of source control (see `.gitignore`) —
each contributor/environment must place the dataset here locally, or point `config.yaml` at another location.

- `raw/Training/` — original Training folder from the dataset (used for both training and validation, per config split).
- `raw/Testing/` — original Testing folder from the dataset, kept untouched for final evaluation only.

Note: this folder was missing from the original tree and has been added since the dataset needs an explicit,
git-ignored home on disk.

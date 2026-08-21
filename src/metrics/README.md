# src/metrics

One file per evaluation metric, in the project's priority order:

1. `confusion_matrix.py` — Confusion Matrix
2. `recall.py` — Recall (Sensitivity)
3. `f1_score.py` — F1-Score (Macro)
4. `precision.py` — Precision
5. `roc_auc.py` — ROC-AUC (Macro / One-vs-Rest)
6. `accuracy.py` — Accuracy

Renamed from `PascalCase.py` (and invalid `F1-Score.py` / `ROC-AUC.py`) to valid, PEP8-compliant `snake_case.py`
module names.

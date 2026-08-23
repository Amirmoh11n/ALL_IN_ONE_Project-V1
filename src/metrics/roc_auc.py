"""
ROC-AUC (Macro / One-vs-Rest) metric (priority #5). Wraps scikit-learn's
implementation.

Unlike the other metrics (which need only predicted class labels), ROC-AUC
needs predicted *probabilities* per class, since it measures ranking quality
across all classification thresholds, not just the final argmax decision.
"""

from typing import List, Optional, Sequence

from sklearn.metrics import roc_auc_score


class ROCAUCMetric:
    """Computes ROC-AUC, macro-averaged, one-vs-rest, for multi-class predictions."""

    @staticmethod
    def compute(
        y_true: List[int],
        y_score: Sequence[Sequence[float]],
        num_classes: Optional[int] = None,
        average: str = "macro",
    ) -> float:
        """
        Args:
            y_true: Ground-truth integer class labels.
            y_score: Predicted probabilities, shape (n_samples, num_classes)
                (e.g. softmax output of the model -- NOT raw logits, and NOT
                the argmax predicted label).
            num_classes: If given, passed as the explicit label set so the
                score is well-defined even if a class is missing from y_true.
            average: sklearn averaging strategy. "macro" (unweighted mean
                across classes) is the project default.

        Returns:
            The macro-averaged, one-vs-rest ROC-AUC as a float.
        """
        labels = list(range(num_classes)) if num_classes is not None else None
        return roc_auc_score(
            y_true, y_score, multi_class="ovr", average=average, labels=labels,
        )

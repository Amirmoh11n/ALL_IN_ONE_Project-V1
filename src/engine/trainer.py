"""
Training loop orchestration for EfficientNet-B3: CrossEntropyLoss (optionally
class-weighted, computed at runtime from the actual training split),
Adam optimizer, ReduceLROnPlateau LR scheduling, early stopping on validation
loss, best-checkpoint saving, and optional MLflow experiment tracking.

All hyperparameters are read from configs/config.yaml (never hardcoded).
"""

import logging
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from a list of integer labels.

    Weight for class i = total_samples / (num_classes * count(class i)).
    Classes with more samples get a smaller weight; classes with fewer
    samples get a larger weight, so CrossEntropyLoss penalizes minority-class
    mistakes more heavily.

    Args:
        labels: List of integer class labels (e.g. from the training split).
        num_classes: Total number of classes.

    Returns:
        A float tensor of shape (num_classes,).
    """
    counts = Counter(labels)
    total = len(labels)
    weights = [
        total / (num_classes * counts[i]) if counts.get(i, 0) > 0 else 0.0
        for i in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """Stops training when a monitored metric stops improving for `patience` epochs."""

    def __init__(self, patience: int = 5, mode: str = "min") -> None:
        """
        Args:
            patience: Number of epochs with no improvement before stopping.
            mode: "min" if lower is better (e.g. loss), "max" if higher is better.
        """
        self.patience = patience
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score: float) -> None:
        """Update internal state with the latest epoch's monitored score."""
        if self.best_score is None or self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def _is_improvement(self, current_score: float) -> bool:
        if self.mode == "min":
            return current_score < self.best_score
        return current_score > self.best_score


class Trainer:
    """Orchestrates the EfficientNet-B3 training loop."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ConfigLoader,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            model: The (uncompiled) classifier, e.g. EfficientNetB3Classifier.
            train_loader: Training DataLoader (from DataPipeline).
            val_loader: Validation DataLoader (from DataPipeline).
            config: Loaded ConfigLoader over configs/config.yaml.
            checkpoint_dir: Where to save the best checkpoint. Defaults to
                "artifacts/checkpoints" (the project's fixed artifacts location).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        requested_device = config.get("training.device", "cuda")
        self.device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        class_weights = None
        if config.get("training.use_class_weights", False):
            class_weights = self._compute_class_weights_from_train_loader().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        lr = config.get("training.learning_rate", 1e-4)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.get("training.lr_scheduler.factor", 0.5),
            patience=config.get("training.lr_scheduler.patience", 2),
        )

        self.early_stopping = EarlyStopping(
            patience=config.get("training.early_stopping.patience", 5),
            mode="min",
        )

        self.checkpoint_dir = Path(checkpoint_dir or "artifacts/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

        self.mlflow_enabled = config.get("tracking.mlflow.enabled", False)

    def _compute_class_weights_from_train_loader(self) -> torch.Tensor:
        """Derive class weights from the actual samples in train_loader.dataset."""
        labels = [label for _, label in self.train_loader.dataset.samples]
        num_classes = len(set(labels))
        weights = compute_class_weights(labels, num_classes)
        logger.info("Computed class weights from training split: %s", weights.tolist())
        return weights

    def fit(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Run the full training loop with early stopping.

        Args:
            num_epochs: Max epochs to run. Defaults to config's training.epochs.

        Returns:
            History dict with "train_loss", "val_loss", "val_accuracy" per epoch.
        """
        num_epochs = num_epochs or self.config.get("training.epochs", 30)
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        run_context = self._mlflow_run_context()
        with run_context:
            self._log_mlflow_params(num_epochs)
            for epoch in range(1, num_epochs + 1):
                train_loss = self._train_one_epoch()
                val_loss, val_accuracy = self._validate_one_epoch()
                self.scheduler.step(val_loss)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)

                logger.info(
                    "Epoch %d/%d - train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch, num_epochs, train_loss, val_loss, val_accuracy,
                )
                self._log_mlflow_metrics(epoch, train_loss, val_loss, val_accuracy)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)

                self.early_stopping.step(val_loss)
                if self.early_stopping.should_stop:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        return history

    def _train_one_epoch(self) -> float:
        """Run one training epoch. Returns the mean training loss."""
        self.model.train()
        running_loss = 0.0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
        return running_loss / len(self.train_loader.dataset)

    def _validate_one_epoch(self) -> "tuple[float, float]":
        """Run one validation epoch. Returns (mean val loss, val accuracy)."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
        val_loss = running_loss / len(self.val_loader.dataset)
        val_accuracy = correct / len(self.val_loader.dataset)
        return val_loss, val_accuracy

    def _save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        """Save the current model state as the new best checkpoint."""
        checkpoint_path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict(), "val_loss": val_loss},
            checkpoint_path,
        )
        logger.info("Saved new best checkpoint (val_loss=%.4f) to %s", val_loss, checkpoint_path)
        return checkpoint_path

    def _mlflow_run_context(self):
        """Return an MLflow run context manager if enabled, else a no-op context."""
        if not self.mlflow_enabled:
            return nullcontext()
        import mlflow

        mlflow.set_tracking_uri(self.config.get("tracking.mlflow.tracking_uri", "artifacts/mlruns"))
        mlflow.set_experiment(self.config.get("tracking.mlflow.experiment_name", "brain_tumor_classification"))
        return mlflow.start_run()

    def _log_mlflow_params(self, num_epochs: int) -> None:
        if not self.mlflow_enabled:
            return
        import mlflow

        mlflow.log_params({
            "learning_rate": self.config.get("training.learning_rate", 1e-4),
            "num_epochs": num_epochs,
            "batch_size": self.train_loader.batch_size,
            "use_class_weights": self.config.get("training.use_class_weights", False),
            "device": str(self.device),
        })

    def _log_mlflow_metrics(self, epoch: int, train_loss: float, val_loss: float, val_accuracy: float) -> None:
        if not self.mlflow_enabled:
            return
        import mlflow

        mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy},
            step=epoch,
        )

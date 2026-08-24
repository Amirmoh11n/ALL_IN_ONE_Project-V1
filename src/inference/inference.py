"""
Inference pipeline: loads a trained checkpoint and classifies a single MRI
image, returning the predicted tumor class and per-class confidence scores.
This is what the FastAPI backend (webapplication/backend) will call when a
user uploads an image.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.augment import AugmentationFactory
from src.data.classes import TumorClasses
from src.models.efficientnet import EfficientNetB3Classifier
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """A single image's prediction: the winning class plus the full probability
    distribution over all classes."""

    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict:
        """Flat, JSON-friendly dict representation (e.g. for the FastAPI response)."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class InferencePipeline:
    """Loads a trained checkpoint once, then classifies MRI images on demand."""

    def __init__(
        self,
        checkpoint_path: Path,
        config: ConfigLoader,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            checkpoint_path: Path to a checkpoint saved by Trainer
                (e.g. artifacts/checkpoints/best_model.pt).
            config: Loaded ConfigLoader over configs/config.yaml (reads
                model.num_classes, data.image_size, data.normalization).
            device: Device to run inference on. Defaults to CUDA if available, else CPU.
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = config.get("model.num_classes", TumorClasses.num_classes())
        model = EfficientNetB3Classifier(num_classes=num_classes, pretrained=False)
        self.model = load_model_checkpoint(model, checkpoint_path, device=self.device)

        image_size = config.get("data.image_size", 300)
        mean = config.get("data.normalization.mean")
        std = config.get("data.normalization.std")
        self.transform = AugmentationFactory.build_eval_transforms(image_size, mean, std)

        logger.info("InferencePipeline ready (checkpoint=%s, device=%s).", checkpoint_path, self.device)

    def predict(self, image: Union[str, Path, Image.Image]) -> PredictionResult:
        """Classify a single image.

        Args:
            image: A file path/str, or an already-loaded PIL Image.

        Returns:
            A PredictionResult with the predicted class name, its confidence,
            and the full per-class probability distribution.
        """
        pil_image = self._to_pil_image(image)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        predicted_index = int(probabilities.argmax().item())
        probability_dict = {
            TumorClasses.index_to_name(i): float(probabilities[i].item())
            for i in range(len(probabilities))
        }

        return PredictionResult(
            predicted_class=TumorClasses.index_to_name(predicted_index),
            confidence=float(probabilities[predicted_index].item()),
            probabilities=probability_dict,
        )

    @staticmethod
    def _to_pil_image(image: Union[str, Path, Image.Image]) -> Image.Image:
        """Normalize the input into an RGB PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

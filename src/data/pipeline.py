"""
DataPipeline: a thin facade that wires together DatasetDownloader, DatasetSplitter,
AugmentationFactory, and BrainTumorDataset into ready-to-use PyTorch DataLoaders.

This is a new file (not present in the originally proposed tree), added so callers
(e.g. the training engine) have a single entry point instead of re-wiring the
individual data components themselves. Reads all settings from configs/config.yaml.
"""

import logging
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

from src.data.augment import AugmentationFactory
from src.data.dataset import BrainTumorDataset
from src.data.downloader import DatasetDownloader
from src.data.splitter import DatasetSplitter
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class DataPipeline:
    """Prepares train/val/test DataLoaders for the Brain Tumor MRI Dataset.

    Steps performed:
        1. Ensure the raw dataset exists locally (download via kagglehub if missing).
        2. Stratified split of the Training folder into train / validation subsets.
        3. Build BrainTumorDataset instances with the appropriate transforms
           (augmentation for train; normalization-only for val/test).
        4. Wrap each in a DataLoader (train shuffled, val/test not shuffled).
    """

    def __init__(self, config: ConfigLoader) -> None:
        """
        Args:
            config: A loaded ConfigLoader over configs/config.yaml.
        """
        self.config = config

    def prepare(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Run the full pipeline and return (train_loader, val_loader, test_loader)."""
        raw_dir = Path(self.config.get("data.raw_dir", "data/raw"))
        train_dir = raw_dir / self.config.get("data.train_dir_name", "Training")
        test_dir = raw_dir / self.config.get("data.test_dir_name", "Testing")
        kaggle_handle = self.config.get("data.kaggle.dataset_slug")

        downloader = DatasetDownloader(kaggle_handle=kaggle_handle, train_dir=train_dir, test_dir=test_dir)
        downloader.ensure_dataset()

        val_ratio = self.config.get("data.val_split", 0.15)
        random_seed = self.config.get("data.seed", 42)
        splitter = DatasetSplitter(train_dir=train_dir, val_ratio=val_ratio, random_seed=random_seed)
        split_result = splitter.split()

        image_size = self.config.get("data.image_size", 300)
        mean = self.config.get("data.normalization.mean")
        std = self.config.get("data.normalization.std")

        train_transform = self._build_train_transform(image_size, mean, std)
        eval_transform = AugmentationFactory.build_eval_transforms(image_size, mean, std)

        train_dataset = BrainTumorDataset(split_result.train_samples, transform=train_transform)
        val_dataset = BrainTumorDataset(split_result.val_samples, transform=eval_transform)
        test_dataset = BrainTumorDataset.from_directory(test_dir, transform=eval_transform)

        batch_size = self.config.get("data.dataloader.batch_size", 32)
        num_workers = self.config.get("data.dataloader.num_workers", 4)
        shuffle_train = self.config.get("data.dataloader.shuffle_train", True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )

        logger.info(
            "DataPipeline ready: %d train / %d val / %d test samples.",
            len(train_dataset), len(val_dataset), len(test_dataset),
        )
        return train_loader, val_loader, test_loader

    def _build_train_transform(self, image_size: int, mean, std):
        """Read augmentation params from config and build the training transform pipeline."""
        flip_p = self.config.get("data.augmentation.random_horizontal_flip_p", 0.5)
        rotation_degrees = self.config.get("data.augmentation.random_rotation_degrees", 15)
        brightness = self.config.get("data.augmentation.color_jitter.brightness", 0.1)
        contrast = self.config.get("data.augmentation.color_jitter.contrast", 0.1)
        return AugmentationFactory.build_train_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            flip_p=flip_p,
            rotation_degrees=rotation_degrees,
            brightness=brightness,
            contrast=contrast,
        )

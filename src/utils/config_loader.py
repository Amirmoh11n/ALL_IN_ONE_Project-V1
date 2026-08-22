"""
Loads and provides access to configs/config.yaml, the single source of truth
for experiment settings (data paths, split ratios, image/augmentation settings,
training hyperparameters, etc.) so experiments can be run without touching source code.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Loads a YAML config file into a plain dict, with a dotted-path getter."""

    def __init__(self, config_path: Path) -> None:
        """
        Args:
            config_path: Path to a YAML config file (e.g. configs/config.yaml).
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Read and parse the YAML file. Returns an empty dict for an empty file."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def raw(self) -> Dict[str, Any]:
        """Return the full config as a plain dict."""
        return self._config

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a nested config value using a dotted path.

        Example:
            config.get("data.split.val_ratio", 0.15)

        Args:
            dotted_key: Dot-separated path into the nested config dict.
            default: Value returned if the path does not exist.
        """
        node: Any = self._config
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

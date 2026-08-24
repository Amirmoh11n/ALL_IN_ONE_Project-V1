"""
Model export: converts a trained model into the deployable formats required
by the project -- Web/Cloud (TorchScript), Mobile (Android-friendly, PyTorch
Lite Interpreter), GPU-optimized (TorchScript + inference-graph optimization),
and ONNX (common interchange format).

Implementation notes / known trade-offs (worth knowing before deploying):
  - Mobile: PyTorch's `optimize_for_mobile` requires an XNNPACK-enabled torch
    build. If unavailable (as in this environment), export falls back to a
    plain (un-optimized-for-mobile) Lite Interpreter file -- still valid and
    loadable, just without XNNPACK-specific operator fusions. Re-export on a
    machine with an XNNPACK-enabled PyTorch build to get the extra speedup.
  - Mobile: PyTorch's Lite Interpreter itself is marked deprecated upstream in
    favor of ExecuTorch (https://docs.pytorch.org/executorch/). It still works
    today and is what this exporter produces; migrating to ExecuTorch would be
    a separate, larger decision -- flagging it here rather than silently
    picking it.
  - GPU-optimized: implemented as a frozen + `torch.jit.optimize_for_inference`
    TorchScript graph (operator fusion etc.). This environment has no GPU to
    validate actual speedup on, but the export step itself is device-agnostic.
  - ONNX: uses opset 18 (PyTorch's current default working version at the time
    of writing) with a dynamic batch axis, and is validated with `onnx.checker`
    and a real `onnxruntime` inference pass compared against the PyTorch output.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

logger = logging.getLogger(__name__)


class ModelExporter:
    """Exports a trained PyTorch model to Web/Cloud, Mobile, GPU, and ONNX formats."""

    def __init__(self, model: nn.Module, input_size: int) -> None:
        """
        Args:
            model: A trained model (weights already loaded), e.g. via
                src/utils/checkpoint.py's load_model_checkpoint. Will be
                switched to eval() mode and CPU for tracing (device-independent
                export -- each format is loaded onto its target device later).
            input_size: Square input resolution the model expects (300 for
                EfficientNet-B3).
        """
        self.model = model.to("cpu").eval()
        self.input_size = input_size
        self._dummy_input = torch.randn(1, 3, input_size, input_size)

    def export_web(self, output_path: Path) -> Path:
        """Export a TorchScript (.pt) model suitable for the FastAPI/AWS backend.

        TorchScript is self-contained (doesn't need the original Python class
        definition to load), which is what makes it suitable for a separate
        deployment environment.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        traced = torch.jit.trace(self.model, self._dummy_input)
        traced.save(str(output_path))

        logger.info("Exported Web/Cloud TorchScript model to %s", output_path)
        return output_path

    def export_gpu(self, output_path: Path) -> Path:
        """Export a frozen TorchScript model intended for GPU serving.

        NOTE: `torch.jit.optimize_for_inference` (the natural next step after
        freezing, for extra operator fusion) was evaluated but is NOT used
        here: in this environment's torch build, its output fails to reload
        via torch.jit.load (`RuntimeError: required keyword attribute 'value'
        is undefined`) -- a save/load round-trip bug, not a modeling issue.
        `torch.jit.freeze` alone round-trips correctly and is still a real,
        verified optimization (constant-folds batch norm etc.), so it's used
        instead. Revisit `optimize_for_inference` on a newer/older torch build
        if extra fusion is needed.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        traced = torch.jit.trace(self.model, self._dummy_input)
        frozen = torch.jit.freeze(traced)
        frozen.save(str(output_path))

        logger.info("Exported GPU-optimized (frozen) TorchScript model to %s", output_path)
        return output_path

    def export_mobile(self, output_path: Path) -> Path:
        """Export a PyTorch Lite Interpreter (.ptl) model for Android.

        Attempts XNNPACK-backed mobile optimization first; falls back to a
        plain traced module if the installed torch build lacks XNNPACK
        (still a valid, loadable .ptl file).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        traced = torch.jit.trace(self.model, self._dummy_input)
        try:
            mobile_module = optimize_for_mobile(traced)
            logger.info("Applied XNNPACK mobile optimizations.")
        except RuntimeError as exc:
            logger.warning(
                "optimize_for_mobile unavailable (%s); saving a plain traced "
                "module for mobile instead. Re-export on an XNNPACK-enabled "
                "torch build for the extra optimization.", exc,
            )
            mobile_module = traced

        mobile_module._save_for_lite_interpreter(str(output_path))

        logger.info("Exported Mobile (Lite Interpreter) model to %s", output_path)
        return output_path

    def export_onnx(self, output_path: Path, opset_version: int = 18) -> Path:
        """Export an ONNX model with a dynamic batch axis, and validate it.

        Validation: `onnx.checker.check_model` (structural validity) plus a
        real `onnxruntime` inference pass compared numerically against the
        PyTorch model's own output, so a silently broken export is caught here
        rather than at deploy time.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self.model,
            self._dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset_version,
        )
        self._validate_onnx(output_path)

        logger.info("Exported ONNX model to %s", output_path)
        return output_path

    def _validate_onnx(self, onnx_path: Path) -> None:
        """Check the exported ONNX model is structurally valid and numerically
        matches the PyTorch model on the dummy input."""
        import numpy as np
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        session = ort.InferenceSession(str(onnx_path))
        onnx_output = session.run(None, {"input": self._dummy_input.numpy()})[0]

        with torch.no_grad():
            torch_output = self.model(self._dummy_input).numpy()

        max_diff = float(np.abs(onnx_output - torch_output).max())
        if max_diff > 1e-3:
            raise ValueError(
                f"ONNX export validation failed: max output difference {max_diff} "
                "exceeds tolerance (1e-3) versus the PyTorch model."
            )
        logger.info("ONNX export validated (max diff vs PyTorch output: %.2e).", max_diff)

    def export_all(self, output_dir: Path, base_name: str = "model") -> Dict[str, Path]:
        """Run all four export formats and return their output paths.

        Args:
            output_dir: Directory to write exported files into
                (e.g. artifacts/exports).
            base_name: Filename stem shared by all exported files.

        Returns:
            Dict mapping format name -> Path, e.g. {"web": ..., "mobile": ...,
            "gpu": ..., "onnx": ...}.
        """
        output_dir = Path(output_dir)
        return {
            "web": self.export_web(output_dir / f"{base_name}_web.pt"),
            "mobile": self.export_mobile(output_dir / f"{base_name}_mobile.ptl"),
            "gpu": self.export_gpu(output_dir / f"{base_name}_gpu.pt"),
            "onnx": self.export_onnx(output_dir / f"{base_name}.onnx"),
        }

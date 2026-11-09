from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from .yolo_decode import decode_yolo_output, letterbox
from .schema import Det

logger = logging.getLogger(__name__)

class OrtDetector:
    """Load a YOLO .onnx model and run detection on BGR frames."""
    def __init__(self, model_path: str | Path, device: str = "cpu",
                 imgsz: int = 640, confidence: float = 0.35,
                 names: list[str] | None = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime") from exc
        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.path}")
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.names = list(names) if names else None
        providers = ["CPUExecutionProvider"]
        if device.lower() not in {"cpu", ""}:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                logger.warning("CUDA EP unavailable; using CPU ONNX Runtime")
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs")
        self.input_name = inputs[0].name
        shape = inputs[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.imgsz = int(shape[2])
        outputs = self.session.get_outputs()
        self.output_name = outputs[0].name if outputs else None
        logger.info("ORT model=%s providers=%s", self.path, self.session.get_providers())

    def predict(self, frame: np.ndarray) -> list[Det]:
        """Detect objects on one BGR frame."""
        nchw, scale, pad_x, pad_y = letterbox(frame, self.imgsz)
        feeds = {self.input_name: nchw}
        outs = self.session.run([self.output_name], feeds) if self.output_name else self.session.run(None, feeds)
        return decode_yolo_output(
            outs[0], scale, pad_x, pad_y, confidence=self.confidence,
            names=self.names, frame_shape=frame.shape
        )

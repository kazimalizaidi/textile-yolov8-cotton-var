"""Thin wrapper turning a trained YOLOv8 weights file into the (N, 6) raw arrays the ingestor expects."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("textile.detector")


class YoloV8Detector:
    """Runs inference on one image and returns [x1, y1, x2, y2, conf, cls] in original pixel units.

    conf defaults low (0.25) on purpose: final confidence filtering and NMS are done by
    DetectionIngestor so that every rejection is counted and logged in one place."""

    def __init__(self, weights: Path, conf: float = 0.25, imgsz: int = 640, device: Optional[str] = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("pip install ultralytics to use the YOLO detector") from exc
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Weights file not found: {weights}")
        self._model = YOLO(str(weights))
        self._conf, self._imgsz, self._device = conf, imgsz, device
        LOGGER.info("Loaded YOLOv8 weights %s (classes: %s)", weights, self._model.names)

    @property
    def class_names(self) -> dict:
        return dict(self._model.names)

    def detect(self, image_path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
        result = self._model.predict(source=str(image_path), conf=self._conf, imgsz=self._imgsz,
                                     device=self._device, verbose=False)[0]
        raw = result.boxes.data.cpu().numpy().astype(np.float32).reshape(-1, 6)
        height, width = result.orig_shape
        return raw, (int(width), int(height))

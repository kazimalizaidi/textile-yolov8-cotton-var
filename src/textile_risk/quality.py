"""Module 1 - Deep Learning Quality Engine (YOLOv8 ingestion + batch scoring)."""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("textile.quality")


class DefectType(IntEnum):
    """Class ids match the YOLOv8 model's label map."""

    HOLE = 0
    OIL_STAIN = 1
    MISSING_THREAD = 2


# Figshare "Fabric Defects Object Detection" dataset: native model class ids (confirmed by visual
# inspection) and how they are mapped onto this project's defect types. Cutting and crack are both
# linear/severed-yarn style defects, so they map to MISSING_THREAD. Change this mapping to taste.
FIGSHARE_CLASS_NAMES: Dict[int, str] = {0: "oil_stain", 1: "cutting", 2: "crack", 3: "hole"}
FIGSHARE_CLASS_MAP: Dict[int, "Optional[DefectType]"] = {
    0: DefectType.OIL_STAIN, 1: DefectType.MISSING_THREAD, 2: DefectType.MISSING_THREAD, 3: DefectType.HOLE}


def load_box_limits(path: "Path") -> "Dict[DefectType, float]":
    """Read the by_defect_type section of a box_limits.json (see: textile-risk-train box-limits)."""
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {DefectType[name]: float(v) for name, v in data["by_defect_type"].items()}


def parse_class_map(text: str) -> "Dict[int, Optional[DefectType]]":
    """'figshare' or '0:oil_stain,1:missing_thread,2:ignore,3:hole' -> {model_class_id: DefectType|None}."""
    if text.strip().lower() == "figshare":
        return dict(FIGSHARE_CLASS_MAP)
    out: Dict[int, Optional[DefectType]] = {}
    for item in text.split(","):
        cid, _, name = item.partition(":")
        name = name.strip().upper()
        if not cid.strip().isdigit() or not name:
            raise ValueError(f"Bad class-map entry '{item}'")
        if name == "IGNORE":
            out[int(cid)] = None
        elif name in DefectType.__members__:
            out[int(cid)] = DefectType[name]
        else:
            raise ValueError(f"Unknown defect type '{name}' (use {list(DefectType.__members__)} or ignore)")
    return out


@dataclass(frozen=True)
class FrameGeometry:
    """Camera frame geometry. One frame = frame_length_m of fabric along the roll."""

    width_px: int = 1600
    height_px: int = 1000
    fabric_width_m: float = 1.6
    frame_length_m: float = 1.0

    @property
    def px_per_m_x(self) -> float:
        return self.width_px / self.fabric_width_m

    @property
    def px_per_m_y(self) -> float:
        return self.height_px / self.frame_length_m


@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def iou(self, other: "BoundingBox") -> float:
        ix = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        iy = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class QualityConfig:
    """All tunables of the quality engine (explicit and auditable)."""

    type_weights: Mapping[DefectType, float] = field(
        default_factory=lambda: {
            DefectType.HOLE: 1.0,
            DefectType.MISSING_THREAD: 0.7,
            DefectType.OIL_STAIN: 0.5,
        }
    )
    reference_area_m2: float = 0.0025  # 25 cm^2: area at which size doubles severity
    severe_threshold: float = 2.0  # severity >= this => "severe" defect
    min_confidence: float = 0.25
    nms_iou: float = 0.50
    kappa: float = 0.35  # decay rate of the quality score vs weighted density
    critical_score: float = 70.0  # below this the batch is down-valued
    price_floor_multiplier: float = 0.40
    price_elasticity: float = 2.0
    base_price_per_m_usd: float = 3.20


@dataclass(frozen=True)
class Detection:
    batch_id: str
    frame_id: int
    defect_type: DefectType
    confidence: float
    bbox: BoundingBox
    area_m2: float
    severity: float
    is_severe: bool


class YoloV8Simulator:
    """
    Generates raw detections in the exact layout of Ultralytics `boxes.data`:
    float32 array of shape (N, 6) = [x1, y1, x2, y2, confidence, class_id].
    Includes duplicates (to exercise NMS) and sub-threshold false positives.
    """

    def __init__(self, geometry: FrameGeometry, seed: int) -> None:
        self._geo = geometry
        self._rng = np.random.default_rng(seed)

    def _box(self, cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def _random_defect(self) -> List[float]:
        rng = self._rng
        cls = int(rng.choice([0, 1, 2], p=[0.35, 0.40, 0.25]))
        cx = float(rng.uniform(0, self._geo.width_px))
        cy = float(rng.uniform(0, self._geo.height_px))
        conf = float(np.clip(rng.beta(6.0, 2.0), 0.0, 1.0))
        if cls == DefectType.HOLE:
            s = float(rng.lognormal(np.log(45.0), 0.5))
            w, h = s, s * float(rng.uniform(0.7, 1.3))
        elif cls == DefectType.OIL_STAIN:
            s = float(rng.lognormal(np.log(70.0), 0.5))
            w, h = s * float(rng.uniform(0.8, 1.4)), s * float(rng.uniform(0.8, 1.4))
        else:  # missing thread: long thin box, warp or weft direction
            length = float(rng.lognormal(np.log(250.0), 0.5))
            thick = float(rng.uniform(3.0, 10.0))
            w, h = (thick, length) if rng.random() < 0.5 else (length, thick)
        return [*self._box(cx, cy, w, h), conf, float(cls)]

    def _severe_hole(self) -> List[float]:
        rng = self._rng
        s = float(rng.uniform(120.0, 180.0))
        cx = float(rng.uniform(s, self._geo.width_px - s))
        cy = float(rng.uniform(s, self._geo.height_px - s))
        conf = float(rng.uniform(0.90, 0.98))
        return [*self._box(cx, cy, s, s), conf, float(DefectType.HOLE)]

    def generate_frame(self, defect_rate: float, force_severe: bool = False) -> np.ndarray:
        rng = self._rng
        rows: List[List[float]] = [self._random_defect() for _ in range(int(rng.poisson(defect_rate)))]
        if force_severe:
            rows.append(self._severe_hole())
        duplicates: List[List[float]] = []
        for row in rows:
            if rng.random() < 0.15:  # model emitted a near-duplicate box
                jitter = rng.normal(0.0, 2.0, size=4)
                duplicates.append([row[0] + jitter[0], row[1] + jitter[1],
                                   row[2] + jitter[2], row[3] + jitter[3], row[4] * 0.8, row[5]])
        rows.extend(duplicates)
        if rng.random() < 0.05:  # low-confidence false positive
            fp = self._random_defect()
            fp[4] = float(rng.uniform(0.05, 0.20))
            rows.append(fp)
        return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


class DetectionIngestor:
    """Validates, cleans and enriches raw YOLOv8 output into typed Detection objects."""

    def __init__(self, geometry: FrameGeometry, config: QualityConfig,
                 class_map: Optional[Mapping[int, Optional[DefectType]]] = None,
                 max_area_fraction: Optional[Mapping[DefectType, float]] = None) -> None:
        self._geo = geometry
        self._cfg = config
        # largest plausible box (fraction of the frame) per defect type, derived from labelled data
        self._max_area: Dict[DefectType, float] = dict(max_area_fraction or {})
        # model class id -> DefectType (None = ignore). Default: identity over DefectType ids.
        self._class_map: Dict[int, Optional[DefectType]] = (
            dict(class_map) if class_map is not None else {int(d): d for d in DefectType})
        self.rejections: Counter[str] = Counter()

    def ingest(self, batch_id: str, frame_id: int, raw: np.ndarray) -> List[Detection]:
        if raw.ndim != 2 or raw.shape[1] != 6:
            raise ValueError(f"Expected raw detections of shape (N, 6), got {raw.shape}")
        candidates: List[Tuple[DefectType, float, BoundingBox]] = []
        for x1, y1, x2, y2, conf, cls in raw.astype(np.float64):
            if not np.all(np.isfinite([x1, y1, x2, y2, conf, cls])):
                self.rejections["non_finite"] += 1
                continue
            if cls != int(cls) or int(cls) not in self._class_map:
                self.rejections["unknown_class"] += 1
                continue
            mapped = self._class_map[int(cls)]
            if mapped is None:
                self.rejections["ignored_class"] += 1
                continue
            if not 0.0 <= conf <= 1.0:
                self.rejections["confidence_out_of_range"] += 1
                continue
            if conf < self._cfg.min_confidence:
                self.rejections["below_min_confidence"] += 1
                continue
            box = BoundingBox(
                float(np.clip(min(x1, x2), 0.0, self._geo.width_px)),
                float(np.clip(min(y1, y2), 0.0, self._geo.height_px)),
                float(np.clip(max(x1, x2), 0.0, self._geo.width_px)),
                float(np.clip(max(y1, y2), 0.0, self._geo.height_px)),
            )
            if box.width < 2.0 or box.height < 2.0:
                self.rejections["degenerate_box"] += 1
                continue
            limit = self._max_area.get(mapped)
            if limit is not None and box.area / float(self._geo.width_px * self._geo.height_px) > limit:
                self.rejections["oversized_box"] += 1
                continue
            candidates.append((mapped, float(conf), box))

        kept: List[Tuple[DefectType, float, BoundingBox]] = []
        for cand in sorted(candidates, key=lambda c: c[1], reverse=True):  # greedy class-wise NMS
            if any(k[0] == cand[0] and k[2].iou(cand[2]) > self._cfg.nms_iou for k in kept):
                self.rejections["suppressed_by_nms"] += 1
                continue
            kept.append(cand)

        detections: List[Detection] = []
        for defect_type, conf, box in sorted(kept, key=lambda c: (c[2].y_min, c[2].x_min)):
            area_m2 = (box.width / self._geo.px_per_m_x) * (box.height / self._geo.px_per_m_y)
            severity = (self._cfg.type_weights[defect_type] * conf
                        * (1.0 + area_m2 / self._cfg.reference_area_m2))
            detections.append(Detection(
                batch_id=batch_id, frame_id=frame_id, defect_type=defect_type, confidence=conf,
                bbox=box, area_m2=area_m2, severity=severity,
                is_severe=severity >= self._cfg.severe_threshold))
        return detections


@dataclass(frozen=True)
class BatchReport:
    batch_id: str
    inspected_length_m: float
    total_defects: int
    severe_defects: int
    defects_by_type: Dict[str, int]
    defect_density_per_m: float
    weighted_density_per_m: float
    quality_score: float
    grade: str
    critical_threshold: float
    price_multiplier: float
    base_value_usd: float
    adjusted_value_usd: float
    value_loss_usd: float
    downvalued: bool


class BatchQualityScorer:
    """
    Batch Quality Score  Q = 100 * exp(-kappa * D),  D = (sum_i severity_i) / L   [per metre]
    Price multiplier     m = 1                                     if Q >= Q_crit
                         m = max(m_floor, (Q / Q_crit)^elasticity) if Q <  Q_crit
    """

    def __init__(self, config: QualityConfig) -> None:
        self._cfg = config

    def _grade(self, score: float) -> str:
        if score >= 90.0:
            return "A"
        if score >= 80.0:
            return "B"
        if score >= self._cfg.critical_score:
            return "C"
        return "D"

    def score(self, batch_id: str, detections: Sequence[Detection], length_m: float) -> BatchReport:
        if length_m <= 0.0:
            raise ValueError("length_m must be positive")
        cfg = self._cfg
        weighted_density = sum(d.severity for d in detections) / length_m
        quality = 100.0 * math.exp(-cfg.kappa * weighted_density)
        if quality >= cfg.critical_score:
            multiplier = 1.0
        else:
            multiplier = max(cfg.price_floor_multiplier,
                             (quality / cfg.critical_score) ** cfg.price_elasticity)
        base_value = length_m * cfg.base_price_per_m_usd
        adjusted = base_value * multiplier
        by_type = Counter(d.defect_type.name for d in detections)
        report = BatchReport(
            batch_id=batch_id, inspected_length_m=length_m, total_defects=len(detections),
            severe_defects=sum(d.is_severe for d in detections),
            defects_by_type={t.name: int(by_type.get(t.name, 0)) for t in DefectType},
            defect_density_per_m=len(detections) / length_m,
            weighted_density_per_m=weighted_density, quality_score=quality,
            grade=self._grade(quality), critical_threshold=cfg.critical_score,
            price_multiplier=multiplier, base_value_usd=base_value,
            adjusted_value_usd=adjusted, value_loss_usd=base_value - adjusted,
            downvalued=multiplier < 1.0)
        LOGGER.info("Batch %s scored: Q=%.2f grade=%s density=%.3f/m weighted=%.3f/m "
                    "multiplier=%.4f value $%.2f -> $%.2f", batch_id, quality, report.grade,
                    report.defect_density_per_m, weighted_density, multiplier,
                    base_value, adjusted)
        return report


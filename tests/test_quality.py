import math

import numpy as np
import pytest

from textile_risk.quality import (BatchQualityScorer, BoundingBox, DefectType, Detection,
                                  DetectionIngestor, QualityConfig, YoloV8Simulator)


def _row(x1, y1, x2, y2, conf, cls):
    return [x1, y1, x2, y2, conf, cls]


def test_iou_identical_disjoint_and_half():
    a = BoundingBox(0, 0, 10, 10)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(BoundingBox(20, 20, 30, 30)) == 0.0
    assert a.iou(BoundingBox(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_severity_formula_exact(geometry, qcfg):
    # 100x100 px at 1000 px/m => 0.1 m x 0.1 m = 0.01 m^2 ; HOLE weight 1.0
    raw = np.array([_row(100, 100, 200, 200, 0.8, 0)], dtype=np.float32)
    det, = DetectionIngestor(geometry, qcfg).ingest("B", 0, raw)
    expected = 1.0 * float(np.float32(0.8)) * (1 + 0.01 / 0.0025)
    assert det.area_m2 == pytest.approx(0.01)
    assert det.severity == pytest.approx(expected)
    assert det.is_severe  # 4.0 >= 2.0


def test_ingestion_filters(geometry, qcfg):
    ing = DetectionIngestor(geometry, qcfg)
    raw = np.array([
        _row(10, 10, 60, 60, 0.9, 0),            # valid
        _row(11, 11, 61, 61, 0.7, 0),            # duplicate -> NMS
        _row(300, 300, 350, 350, 0.1, 1),        # below min confidence
        _row(np.nan, 0, 10, 10, 0.9, 0),         # non-finite
        _row(0, 0, 50, 50, 0.9, 7),              # unknown class
        _row(0, 0, 50, 50, 1.4, 0),              # confidence out of range
        _row(400, 400, 401, 401, 0.9, 2),        # degenerate
    ], dtype=np.float32)
    dets = ing.ingest("B", 0, raw)
    assert len(dets) == 1 and dets[0].confidence == pytest.approx(0.9)
    assert ing.rejections["suppressed_by_nms"] == 1
    assert ing.rejections["below_min_confidence"] == 1
    assert ing.rejections["non_finite"] == 1
    assert ing.rejections["unknown_class"] == 1
    assert ing.rejections["confidence_out_of_range"] == 1
    assert ing.rejections["degenerate_box"] == 1


def test_ingestion_clips_boxes_and_rejects_bad_shape(geometry, qcfg):
    ing = DetectionIngestor(geometry, qcfg)
    det, = ing.ingest("B", 0, np.array([_row(-50, -50, 40, 40, 0.9, 0)], dtype=np.float32))
    assert (det.bbox.x_min, det.bbox.y_min) == (0.0, 0.0)
    with pytest.raises(ValueError):
        ing.ingest("B", 0, np.zeros((3, 5), dtype=np.float32))


def _det(sev: float) -> Detection:
    return Detection("B", 0, DefectType.HOLE, 0.9, BoundingBox(0, 0, 10, 10), 0.0, sev, sev >= 2.0)


def test_score_is_exact_and_monotonic(qcfg):
    scorer = BatchQualityScorer(qcfg)
    r = scorer.score("B", [_det(1.0)] * 10, length_m=100.0)  # D = 0.1
    assert r.quality_score == pytest.approx(100 * math.exp(-0.35 * 0.1))
    assert r.price_multiplier == 1.0 and not r.downvalued
    worse = scorer.score("B", [_det(1.0)] * 50, length_m=100.0)
    assert worse.quality_score < r.quality_score


def test_downvaluation_formula_and_floor(qcfg):
    scorer = BatchQualityScorer(qcfg)
    # D = ln(100/60)/0.35 -> Q = 60 exactly
    d = math.log(100 / 60) / qcfg.kappa
    r = scorer.score("B", [_det(d * 100)], length_m=100.0)
    assert r.quality_score == pytest.approx(60.0)
    assert r.price_multiplier == pytest.approx((60 / 70) ** 2)
    assert r.adjusted_value_usd == pytest.approx(r.base_value_usd * r.price_multiplier)
    floor = scorer.score("B", [_det(1e6)], length_m=100.0)
    assert floor.price_multiplier == qcfg.price_floor_multiplier


def test_scorer_rejects_nonpositive_length(qcfg):
    with pytest.raises(ValueError):
        BatchQualityScorer(qcfg).score("B", [], 0.0)


def test_simulator_layout_and_determinism(geometry):
    a = YoloV8Simulator(geometry, seed=1).generate_frame(2.0)
    b = YoloV8Simulator(geometry, seed=1).generate_frame(2.0)
    assert a.dtype == np.float32 and a.ndim == 2 and a.shape[1] == 6
    np.testing.assert_array_equal(a, b)

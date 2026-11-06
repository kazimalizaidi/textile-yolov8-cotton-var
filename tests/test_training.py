import json
from pathlib import Path

import numpy as np
import pytest

from textile_risk.datacheck import inspect_yolo_dataset
from textile_risk.quality import (FIGSHARE_CLASS_MAP, BatchQualityScorer, DefectType, DetectionIngestor,
                                  FrameGeometry, QualityConfig, parse_class_map)
from textile_risk.training import prepare_dataset


def _roboflow_like(root: Path) -> None:
    """train/valid/test each with images/ + labels/, a raw unlabeled folder, and one leaked duplicate."""
    spec = {"train": [("t1", b"A", "3 0.5 0.5 0.2 0.2\n"), ("t2", b"B", "0 0.4 0.4 0.1 0.1\n1 0.6 0.6 0.2 0.1\n")],
            "valid": [("v1", b"C", "2 0.5 0.5 0.3 0.3\n"), ("v_dup", b"A", "3 0.5 0.5 0.2 0.2\n")],  # v_dup == t1
            "test": [("s1", b"D", "1 0.5 0.5 0.3 0.2\n")]}
    for split, items in spec.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for stem, data, label in items:
            (root / split / "images" / f"{stem}.jpg").write_bytes(data)
            (root / split / "labels" / f"{stem}.txt").write_text(label)
    (root / "raw").mkdir()
    (root / "raw" / "photo_1.jpg").write_bytes(b"Z")
    (root / "data.yaml").write_text("names: ['0', '1', '2', '3']\n")


def test_prepare_dataset_layout_dedup_and_reconciliation(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "prepared"
    _roboflow_like(src)
    rep = prepare_dataset(src, dst)
    assert rep["labelled_pairs_found"] == 5 and rep["images_kept"] == 4
    assert [d["image"] for d in rep["dropped_duplicates"]] == ["v_dup"]
    assert rep["dropped_duplicates"][0]["identical_to"] == "train/t1"
    assert rep["unlabeled_or_unsplit_images_ignored"] == 1
    assert rep["box_reconciliation"]["source"] == {"3": 2, "0": 1, "1": 2, "2": 1} or \
        rep["box_reconciliation"]["source"] == {3: 2, 0: 1, 1: 2, 2: 1}
    assert rep["per_split"]["train"]["boxes_by_class"] == {"oil_stain": 1, "cutting": 1, "crack": 0, "hole": 1}
    assert (dst / "images" / "val" / "v1.jpg").exists() and not (dst / "images" / "val" / "v_dup.jpg").exists()
    chk = inspect_yolo_dataset(dst)
    assert chk.clean and chk.class_counts == {0: 1, 1: 2, 2: 1, 3: 1}
    yaml = (dst / "data.yaml").read_text()
    assert "0: oil_stain" in yaml and "3: hole" in yaml and "val: images/val" in yaml
    assert json.loads((dst / "prepare_report.json").read_text())["images_kept"] == 4


def test_prepare_refuses_nonempty_dst_and_unknown_class(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "prepared"
    _roboflow_like(src)
    prepare_dataset(src, dst)
    with pytest.raises(FileExistsError):
        prepare_dataset(src, dst)
    prepare_dataset(src, dst, overwrite=True)
    (src / "train" / "labels" / "t1.txt").write_text("9 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="class ids"):
        prepare_dataset(src, tmp_path / "other")


def test_prepare_rejects_bad_label_lines(tmp_path):
    src = tmp_path / "src"
    _roboflow_like(src)
    (src / "train" / "labels" / "t1.txt").write_text("3 1.7 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="outside"):
        prepare_dataset(src, tmp_path / "out")


def test_parse_class_map():
    assert parse_class_map("figshare") == FIGSHARE_CLASS_MAP
    m = parse_class_map("0:oil_stain,1:ignore,3:hole")
    assert m == {0: DefectType.OIL_STAIN, 1: None, 3: DefectType.HOLE}
    for bad in ("0:banana", "x:hole", "0"):
        with pytest.raises(ValueError):
            parse_class_map(bad)


def test_ingestor_applies_class_map_ignore_and_merge():
    geo, cfg = FrameGeometry(), QualityConfig()
    ing = DetectionIngestor(geo, cfg, parse_class_map("0:oil_stain,1:missing_thread,2:ignore,3:hole"))
    raw = np.array([[100, 100, 200, 200, 0.9, 3], [300, 300, 400, 400, 0.9, 2], [500, 500, 600, 600, 0.9, 1],
                    [700, 700, 800, 800, 0.9, 5]], dtype=np.float32)
    dets = ing.ingest("B", 0, raw)
    assert sorted(d.defect_type.name for d in dets) == ["HOLE", "MISSING_THREAD"]
    assert ing.rejections["ignored_class"] == 1 and ing.rejections["unknown_class"] == 1


def test_image_folder_pipeline_with_fake_detector(tmp_path, monkeypatch):
    """No ultralytics needed: a stub detector returns three severe holes in a row, then nothing."""
    import textile_risk.detector as det_mod
    from textile_risk.orchestrator import PlatformConfig, TextileRiskPlatform

    class FakeDetector:
        class_names = {0: "oil_stain", 1: "cutting", 2: "crack", 3: "hole"}

        def __init__(self, weights, conf=0.25, imgsz=640, device=None):
            pass

        def detect(self, image_path):
            n = int(Path(image_path).stem)
            rows = [[100, 100, 300, 300, 0.95, 3]] if n < 3 else []  # 200 px square on 1000 px/m => 0.04 m^2
            return np.array(rows, dtype=np.float32).reshape(-1, 6), (1000, 1000)

    monkeypatch.setattr(det_mod, "YoloV8Detector", FakeDetector)
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(6):
        (imgs / f"{i}.jpg").write_bytes(b"x")
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    cfg = PlatformConfig(output_dir=tmp_path / "out", detector="yolo", weights=w, images_dir=imgs,
                         frame_width_m=1.0, frame_length_m=1.0)
    res = TextileRiskPlatform(cfg).run_image_folder("B-T")
    rep = res["batch_reports"][0]
    assert res["images_processed"] == 6 and rep.total_defects == 3 and rep.severe_defects == 3
    assert rep.defects_by_type["HOLE"] == 3 and rep.inspected_length_m == 6.0
    assert len(res["hedge_files"]) == 1  # 3 consecutive severe holes -> exactly one hedge request
    # severity = 1.0 * 0.95 * (1 + 0.04/0.0025) = 16.15 each
    assert rep.weighted_density_per_m == pytest.approx(3 * 0.95 * 17.0 / 6.0, rel=1e-4)


def test_image_folder_requires_complete_class_map(tmp_path, monkeypatch):
    import textile_risk.detector as det_mod
    from textile_risk.orchestrator import PlatformConfig, TextileRiskPlatform

    class Fake:
        class_names = {0: "a", 7: "b"}

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(det_mod, "YoloV8Detector", Fake)
    imgs = tmp_path / "i"
    imgs.mkdir()
    (imgs / "1.jpg").write_bytes(b"x")
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    cfg = PlatformConfig(output_dir=tmp_path / "o", detector="yolo", weights=w, images_dir=imgs,
                         frame_width_m=1.0, frame_length_m=1.0)
    with pytest.raises(ValueError, match="no entry"):
        TextileRiskPlatform(cfg).run_image_folder()


def _dated_dataset(root: Path) -> None:
    """Names follow the figshare pattern: original photo name + Roboflow suffix; three shooting dates."""
    spec = {"train": [("photo_2023-11-19_10-00-00_jpg.rf.aa", b"1"), ("photo_2024-02-10_09-00-00_jpg.rf.bb", b"2"),
                      ("photo_2024-02-11_09-00-05_jpg.rf.cc", b"3")],
            "valid": [("photo_2023-11-19_10-05-00_jpg.rf.dd", b"4"), ("photo_2024-02-11_09-30-00_jpg.rf.ee", b"5")],
            "test": [("photo_2024-02-10_09-00-10_jpg.rf.ff", b"6"), ("photo_2024-02-11_09-00-09_jpg.rf.gg", b"7")]}
    for split, items in spec.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for stem, data in items:
            (root / split / "images" / f"{stem}.jpg").write_bytes(data)
            (root / split / "labels" / f"{stem}.txt").write_text("3 0.5 0.5 0.2 0.2\n")


def test_holdout_date_builds_session_test_set(tmp_path):
    src = tmp_path / "src"
    _dated_dataset(src)
    rep = prepare_dataset(src, tmp_path / "p", holdout_dates=["2023-11-19"])
    assert rep["holdout_dates"] == ["2023-11-19"] and rep["photos_without_timestamp"] == 0
    assert rep["per_split"]["test"]["images_by_date"] == {"2023-11-19": 2}  # train+valid photos of that date
    assert "2023-11-19" not in rep["per_split"]["train"]["images_by_date"]
    assert "2023-11-19" not in rep["per_split"]["val"]["images_by_date"]
    assert rep["per_split"]["train"]["images"] == 4  # 2 orig train (other dates) + 2 orig test moved to train
    assert rep["per_split"]["val"]["images"] == 1
    assert rep["images_kept"] == 7
    assert rep["box_reconciliation"]["source"] == rep["box_reconciliation"]["kept"]


def test_holdout_unknown_date_raises(tmp_path):
    src = tmp_path / "src"
    _dated_dataset(src)
    with pytest.raises(ValueError, match="hold-out dates"):
        prepare_dataset(src, tmp_path / "p", holdout_dates=["2030-01-01"])


def test_sessions_report_gaps(tmp_path):
    from textile_risk.datacheck import inspect_sessions, photo_datetime
    assert photo_datetime("photo_2024-02-11_14-43-15 (2).jpg").second == 15
    assert photo_datetime("photo_2024-02-11_14-43-15_jpg.rf.abc.jpg").minute == 43
    assert photo_datetime("something_else.jpg") is None
    src = tmp_path / "src"
    _dated_dataset(src)
    prepare_dataset(src, tmp_path / "p")
    rep = inspect_sessions(tmp_path / "p")
    assert rep["splits"]["train"]["photos_by_date"] == {"2023-11-19": 1, "2024-02-10": 1, "2024-02-11": 1}
    test = rep["splits"]["test"]
    assert test["within_window"]["10s"] == 2          # test photos are 10 s and 4 s from train photos
    assert rep["splits"]["val"]["within_window"]["10s"] == 0

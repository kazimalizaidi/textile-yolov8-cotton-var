import json
from pathlib import Path

import numpy as np
import pytest

from textile_risk.datacheck import inspect_yolo_dataset
from textile_risk.quality import (FIGSHARE_CLASS_MAP, DefectType, DetectionIngestor,
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


def test_confusion_native_vs_merged_and_summary():
    from textile_risk.training import build_confusion, summarise_confusion
    # classes: 0 oil, 1 cutting, 2 crack, 3 hole.  GT: crack A, hole B, oil C (never detected)
    A, B, C = (10, 10, 60, 60), (100, 100, 150, 150), (200, 200, 260, 260)
    gts = np.array([[2, *A], [3, *B], [0, *C]], dtype=float)
    preds = np.array([[*A, 0.90, 1],                     # crack seen as CUTTING (exact box)
                      [12, 12, 62, 62, 0.50, 2],         # duplicate, labelled crack, slightly shifted
                      [*B, 0.80, 3],                     # hole correct
                      [300, 300, 340, 340, 0.60, 1]],    # false positive (cutting, nothing there)
                     dtype=float)
    native = build_confusion(preds, gts, [0, 1, 2, 3], 4)
    assert native[2, 1] == 1 and native[3, 3] == 1 and native[0, 4] == 1       # crack->cutting, hole ok, oil missed
    assert native[4, 2] == 1 and native[4, 1] == 1                              # unmatched dup (crack) + FP (cutting)
    assert native.sum() == 3 + 2  # 3 GT rows (matched or missed) + 2 false positives
    group_of = [1, 2, 2, 0]  # oil->OIL_STAIN(1), cutting/crack->MISSING_THREAD(2), hole->HOLE(0)
    merged = build_confusion(preds, gts, group_of, 3, nms_iou=0.5)
    assert merged[2, 2] == 1          # crack detected as cutting is correct once merged
    assert merged[3, 2] == 1          # only the real FP remains: the duplicate was removed by NMS
    assert merged[0, 0] == 1 and merged[1, 3] == 1
    s = summarise_confusion(merged, ["HOLE", "OIL_STAIN", "MISSING_THREAD"])
    assert s["MISSING_THREAD"]["recall"] == 1.0 and s["MISSING_THREAD"]["precision"] == 0.5
    assert s["OIL_STAIN"]["recall"] == 0.0 and s["OIL_STAIN"]["missed"] == 1
    sn = summarise_confusion(native, ["oil", "cutting", "crack", "hole"])
    assert sn["crack"]["recall"] == 0.0 and sn["crack"]["detected_as_other_class"] == {"cutting": 1}


def test_confusion_empty_inputs():
    from textile_risk.training import build_confusion
    m = build_confusion(np.zeros((0, 6)), np.zeros((0, 5)), [0, 1], 2)
    assert m.shape == (3, 3) and m.sum() == 0
    m = build_confusion(np.zeros((0, 6)), np.array([[1, 0, 0, 10, 10]], dtype=float), [0, 1], 2)
    assert m[1, 2] == 1 and m.sum() == 1


def test_match_one_to_one_prefers_best_iou_and_reports_leftovers():
    from textile_risk.training import match_one_to_one
    p = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    g = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    pairs, un_g, un_p = match_one_to_one(p, g, 0.5)
    assert pairs == [(0, 0)] and un_g == [1] and un_p == [1, 2]


def test_review_images_stats_and_annotations(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    from textile_risk.training import review_images
    imgs, labs = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    labs.mkdir()
    for n in ("a", "b", "c"):
        Image.new("RGB", (200, 100), (90, 90, 90)).save(imgs / f"{n}.jpg")
        (labs / f"{n}.txt").write_text("2 0.5 0.5 0.5 0.2\n")   # crack, 100 x 20 px
    class Fake:
        def detect(self, path):
            hit = np.array([[50, 40, 150, 60, 0.9, 2]], dtype=np.float32)       # exact crack
            wrong = np.array([[50, 40, 150, 60, 0.8, 1]], dtype=np.float32)     # same box, called cutting
            raw = {"a": hit, "b": np.zeros((0, 6), dtype=np.float32), "c": wrong}[Path(path).stem]
            return raw, (200, 100)
    rep = review_images(Fake(), imgs, labs, 2, tmp_path / "out")
    assert rep["found"]["count"] == 1 and rep["missed_or_wrong_class"]["count"] == 2
    assert rep["problem_boxes_detected_as"] == {"cutting": 1}
    assert rep["found"]["median_short_side_px"] == 20.0 and rep["found"]["median_long_side_px"] == 100.0
    assert rep["found"]["median_area_pct_of_image"] == 10.0
    assert len(rep["annotated_images"]) == 2 and (tmp_path / "out" / "review.json").exists()
    bd = rep["problem_breakdown_by_nearest_prediction"]
    assert bd["no_prediction_in_image"] == 1 and bd["best_iou_at_or_above_threshold(wrong class or box already used)"] == 1


def test_review_breakdown_distinguishes_oversized_boxes_from_silence(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    from textile_risk.training import review_images
    imgs, labs = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    labs.mkdir()
    for n in ("big", "far", "none"):
        Image.new("RGB", (200, 100), (90, 90, 90)).save(imgs / f"{n}.jpg")
        (labs / f"{n}.txt").write_text("2 0.5 0.5 0.1 0.2\n")   # GT 20 x 20 px at (90..110, 40..60)
    class Fake:
        def detect(self, path):
            raw = {"big": np.array([[0, 0, 200, 100, 0.4, 2]], dtype=np.float32),   # oversized: IoU = 400/20000 = 0.02
                   "far": np.array([[0, 0, 20, 20, 0.9, 2]], dtype=np.float32),      # elsewhere: IoU 0
                   "none": np.zeros((0, 6), dtype=np.float32)}[Path(path).stem]
            return raw, (200, 100)
    bd = review_images(Fake(), imgs, labs, 2, tmp_path / "out")["problem_breakdown_by_nearest_prediction"]
    assert bd == {"best_iou_below_0.1": 2, "no_prediction_in_image": 1}


def _label_tree(root: Path) -> Path:
    """prepared-style labels: sizes are normalised w*h; the TEST split holds a huge box that must not count."""
    for split, lines in {"train": ["0 0.5 0.5 0.10 0.10", "1 0.5 0.5 0.2 0.1", "2 0.5 0.5 0.3 0.2", "3 0.5 0.5 0.05 0.05",
                                   "3 0.5 0.5 0.1 0.1"],
                         "val": ["0 0.5 0.5 0.2 0.2", "1 0.5 0.5 0.1 0.1", "2 0.5 0.5 0.1 0.1"],
                         "test": ["3 0.5 0.5 0.9 0.9"]}.items():
        (root / "labels" / split).mkdir(parents=True)
        (root / "labels" / split / "x.txt").write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text("names:\n  0: oil_stain\n  1: cutting\n  2: crack\n  3: hole\n")
    return root / "data.yaml"


def test_compute_box_limits_uses_only_train_val_and_merges_classes(tmp_path):
    from textile_risk.training import compute_box_limits
    rep = compute_box_limits(_label_tree(tmp_path), margin=1.25)
    nat = rep["native"]
    assert nat["hole"]["max"] == pytest.approx(0.01) and nat["hole"]["n"] == 2   # test-split 0.81 box ignored
    assert nat["oil_stain"]["max"] == pytest.approx(0.04) and nat["crack"]["max"] == pytest.approx(0.06)
    assert nat["crack"]["limit"] == pytest.approx(0.075)
    # cutting + crack share MISSING_THREAD -> the larger limit wins
    assert rep["by_defect_type"] == {"OIL_STAIN": pytest.approx(0.05), "MISSING_THREAD": pytest.approx(0.075),
                                     "HOLE": pytest.approx(0.0125)}
    assert rep["splits_used"] == ["train", "val"]


def test_compute_box_limits_requires_boxes_for_every_class(tmp_path):
    from textile_risk.training import compute_box_limits
    yaml = _label_tree(tmp_path)
    (tmp_path / "labels" / "val" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "labels" / "train" / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="No labelled boxes"):
        compute_box_limits(yaml)


def test_ingestor_rejects_oversized_boxes_before_nms():
    geo, cfg = FrameGeometry(width_px=1000, height_px=1000, fabric_width_m=1.0, frame_length_m=1.0), QualityConfig()
    ing = DetectionIngestor(geo, cfg, {0: DefectType.HOLE}, {DefectType.HOLE: 0.05})
    raw = np.array([[100, 100, 400, 400, 0.99, 0],    # 300x300 px = 9% of frame > 5% limit -> rejected
                    [150, 150, 250, 250, 0.60, 0]],   # 1%: would have been suppressed by the big box via NMS
                   dtype=np.float32)
    dets = ing.ingest("B", 0, raw)
    assert len(dets) == 1 and dets[0].confidence == pytest.approx(0.60)
    assert ing.rejections["oversized_box"] == 1 and ing.rejections["suppressed_by_nms"] == 0


def test_confusion_filter_removes_oversized_prediction():
    from textile_risk.training import build_confusion
    gts = np.array([[0, 100, 100, 140, 140]], dtype=float)                      # small true box
    big = np.array([[0, 0, 200, 200, 0.4, 0]], dtype=float)                      # oversized, IoU = 1600/40000 = 0.04
    m = build_confusion(big, gts, [0], 1)
    assert m[0, 1] == 1 and m[1, 0] == 1                                          # one miss + one false positive
    m = build_confusion(big, gts, [0], 1, max_area_frac=[0.05], image_area=200 * 200)
    assert m[0, 1] == 1 and m[1, 0] == 0                                          # false positive removed


def test_pipeline_applies_box_limits_and_reports_rejections(tmp_path, monkeypatch):
    import textile_risk.detector as det_mod
    from textile_risk.orchestrator import PlatformConfig, TextileRiskPlatform

    class Fake:
        class_names = {0: "oil_stain", 1: "cutting", 2: "crack", 3: "hole"}

        def __init__(self, *a, **k):
            pass

        def detect(self, image_path):
            return np.array([[0, 0, 900, 900, 0.9, 3], [100, 100, 150, 150, 0.9, 3]], dtype=np.float32), (1000, 1000)

    monkeypatch.setattr(det_mod, "YoloV8Detector", Fake)
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "1.jpg").write_bytes(b"x")
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"by_defect_type": {"HOLE": 0.1, "OIL_STAIN": 0.2, "MISSING_THREAD": 0.3}}))
    cfg = PlatformConfig(output_dir=tmp_path / "o", detector="yolo", weights=w, images_dir=imgs,
                         frame_width_m=1.0, frame_length_m=1.0, box_limits=limits)
    res = TextileRiskPlatform(cfg).run_image_folder("B-L")
    assert res["batch_reports"][0].total_defects == 1 and res["ingestion_rejections"]["oversized_box"] == 1

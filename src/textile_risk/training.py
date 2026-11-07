"""Dataset preparation, training and evaluation for the figshare fabric-defect dataset (YOLOv8).

    textile-risk-train prepare --src data/defects/figshare_fd --dst data/defects/prepared
    textile-risk-train train   --data data/defects/prepared/data.yaml --epochs 50
    textile-risk-train eval    --weights runs/fabric/weights/best.pt --data data/defects/prepared/data.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .common import atomic_write_json
from .datacheck import IMAGE_EXT, NON_LABEL_STEMS, SPLIT_NAMES, inspect_yolo_dataset, photo_datetime
from .quality import FIGSHARE_CLASS_MAP, FIGSHARE_CLASS_NAMES, DefectType

LOGGER = logging.getLogger("textile.training")
CANON_SPLIT = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}  # earlier split keeps a duplicated image


def _split_of(path: Path) -> str:
    return next((CANON_SPLIT[p.lower()] for p in path.parts if p.lower() in SPLIT_NAMES), "unsplit")


def photo_date(name: str) -> Optional[str]:
    """'photo_2024-02-11_14-43-15_jpg.rf.abc' -> '2024-02-11' (None if the name has no timestamp)."""
    stamp = photo_datetime(name)
    return stamp.date().isoformat() if stamp else None


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_boxes(label: Path) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for i, line in enumerate(label.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{label}:{i}: expected 5 fields, got {len(parts)}")
        cls = float(parts[0])
        if cls != int(cls):
            raise ValueError(f"{label}:{i}: non-integer class '{parts[0]}'")
        vals = [float(v) for v in parts[1:]]
        if not all(0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            raise ValueError(f"{label}:{i}: box outside [0,1] or zero-size")
        out.append((int(cls), line.strip()))
    return out


def prepare_dataset(src: Path, dst: Path, names: Mapping[int, str] = FIGSHARE_CLASS_NAMES,
                    overwrite: bool = False, holdout_dates: Sequence[str] = ()) -> Dict[str, Any]:
    """Copy labelled train/val/test pairs into a clean images/<split> + labels/<split> layout,
    drop val/test images that are byte-identical to an image in an earlier split (leakage),
    write data.yaml and a reconciliation report. Unlabeled raw images are ignored.

    holdout_dates: session hold-out. Every labelled photo taken on one of these dates (parsed from the
    'photo_YYYY-MM-DD_...' filename) becomes the TEST set; original test photos from other dates are moved
    to train; val stays as supplied except that held-out dates never appear in it. This measures
    generalisation to a different shooting session instead of to neighbouring shots of the same session."""
    if not src.is_dir():
        raise FileNotFoundError(f"{src} is not a directory")
    if dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{dst} is not empty (use --overwrite)")
        shutil.rmtree(dst)

    images: Dict[Tuple[str, str], Path] = {}
    labels: Dict[Tuple[str, str], Path] = {}
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        ext, split = p.suffix.lower(), _split_of(p)
        if ext in IMAGE_EXT:
            if (split, p.stem) in images:
                raise ValueError(f"Duplicate image name '{p.stem}' within split '{split}'")
            images[(split, p.stem)] = p
        elif ext == ".txt" and p.stem.lower() not in NON_LABEL_STEMS:
            labels[(split, p.stem)] = p
    holdout = set(holdout_dates)
    unparsed: List[str] = []

    def target_split(k: Tuple[str, str]) -> str:
        if not holdout:
            return k[0]
        d = photo_date(k[1])
        if d is None:
            unparsed.append(k[1])
            return k[0] if k[0] != "test" else "train"
        if d in holdout:
            return "test"
        return "train" if k[0] == "test" else k[0]

    base_pairs = [k for k in images if k in labels and k[0] != "unsplit"]
    target = {k: target_split(k) for k in base_pairs}
    pairs = sorted(base_pairs, key=lambda k: (SPLIT_PRIORITY[target[k]], k[1]))
    if not pairs:
        raise ValueError("No labelled images found inside train/valid/test folders")
    if holdout and not any(target[k] == "test" for k in pairs):
        raise ValueError(f"No labelled photos found for hold-out dates {sorted(holdout)}")

    src_counts: Counter[int] = Counter()
    boxes: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for k in pairs:
        boxes[k] = _read_boxes(labels[k])
        src_counts.update(c for c, _ in boxes[k])
    clashes = [k for k, n in Counter((target[k], k[1]) for k in pairs).items() if n > 1]
    if clashes:
        raise ValueError(f"Image names collide after re-splitting: {clashes[:3]}")
    unknown = sorted(set(src_counts) - set(names))
    if unknown:
        raise ValueError(f"Label files use class ids {unknown} not present in names {dict(names)}")

    seen: Dict[str, Tuple[str, str]] = {}
    kept: List[Tuple[str, str]] = []
    dropped: List[Dict[str, str]] = []
    dropped_counts: Counter[int] = Counter()
    for k in pairs:
        digest = _sha1(images[k])
        if digest in seen:
            dropped.append({"image": k[1], "split": target[k], "identical_to": f"{seen[digest][0]}/{seen[digest][1]}"})
            dropped_counts.update(c for c, _ in boxes[k])
            continue
        seen[digest] = k
        kept.append(k)

    for key in kept:
        split, stem = target[key], key[1]
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        img = images[key]
        shutil.copy2(img, dst / "images" / split / f"{stem}{img.suffix.lower()}")
        (dst / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(line for _, line in boxes[key]) + "\n", encoding="utf-8")

    present = [s for s in ("train", "val", "test") if (dst / "images" / s).is_dir()]
    yaml_lines = [f"path: {dst.resolve().as_posix()}"] + [f"{s}: images/{s}" for s in present] + ["names:"] \
        + [f"  {i}: {n}" for i, n in sorted(names.items())]
    (dst / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    check = inspect_yolo_dataset(dst)
    kept_counts = Counter(check.class_counts)
    if not check.clean:
        raise RuntimeError(f"Prepared dataset failed validation: {check.bad_lines[:3]} {check.images_without_label[:3]}")
    if src_counts != kept_counts + dropped_counts:
        raise RuntimeError(f"Box reconciliation failed: source {dict(src_counts)} != kept {dict(kept_counts)} + dropped {dict(dropped_counts)}")

    per_split: Dict[str, Dict[str, Any]] = {}
    for s in present:
        n_img = sum(1 for k in kept if target[k] == s)
        cc: Counter[int] = Counter(c for k in kept if target[k] == s for c, _ in boxes[k])
        dates = Counter(photo_date(k[1]) or "unknown" for k in kept if target[k] == s)
        per_split[s] = {"images": n_img, "boxes_by_class": {names[c]: cc.get(c, 0) for c in sorted(names)},
                        "images_by_date": dict(sorted(dates.items()))}
    report = {"source": str(src), "destination": str(dst), "class_names": dict(names),
              "labelled_pairs_found": len(pairs), "images_kept": len(kept),
              "dropped_duplicates": dropped, "unlabeled_or_unsplit_images_ignored": len(images) - len(pairs),
              "per_split": per_split, "holdout_dates": sorted(holdout),
              "photos_without_timestamp": len(unparsed),
              "box_reconciliation": {"source": dict(src_counts), "kept": dict(kept_counts), "dropped": dict(dropped_counts)}}
    atomic_write_json(dst / "prepare_report.json", report)
    LOGGER.info("Prepared %d images into %s (%d duplicate(s) dropped)", len(kept), dst, len(dropped))
    return report


def _import_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("pip install ultralytics (or: pip install -e '.[train]')") from exc
    return YOLO


def train_model(data_yaml: Path, weights: str = "yolov8n.pt", epochs: int = 50, imgsz: int = 640,
                batch: int = 16, seed: int = 42, device: Optional[str] = None,
                project: Path = Path("runs"), name: str = "fabric") -> Path:
    """Fine-tune YOLOv8 and return the path of best.pt (plots disabled: backend-only)."""
    model = _import_yolo()(weights)
    model.train(data=str(data_yaml.resolve()), epochs=epochs, imgsz=imgsz, batch=batch, seed=seed,
                deterministic=True, device=device, project=str(project.resolve()), name=name,
                exist_ok=True, plots=False, verbose=False)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished but {best} was not produced")
    LOGGER.info("Best weights: %s", best)
    return best


def evaluate_model(weights: Path, data_yaml: Path, split: str = "test", imgsz: int = 640,
                   device: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate on a held-out split; writes eval_<split>.json beside the weights."""
    metrics = _import_yolo()(str(weights)).val(data=str(data_yaml.resolve()), split=split, imgsz=imgsz,
                                                device=device, plots=False, verbose=False)
    names = metrics.names
    per_class = {names[int(c)]: {"ap50": float(a50), "ap50_95": float(a)}
                 for c, a50, a in zip(metrics.box.ap_class_index, metrics.box.ap50, metrics.box.ap)}
    result = {"weights": str(weights), "split": split, "precision": float(metrics.box.mp),
              "recall": float(metrics.box.mr), "map50": float(metrics.box.map50),
              "map50_95": float(metrics.box.map), "per_class": per_class}
    atomic_write_json(weights.parent / f"eval_{split}.json", result)
    return result


# ----------------------------------------------------------------------------- confusion analysis
def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (P,4) and (G,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def match_one_to_one(p_box: np.ndarray, g_box: np.ndarray, iou_thr: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy one-to-one matching by descending IoU (class-agnostic).
    Returns (matched (pred_idx, gt_idx) pairs, unmatched gt indices, unmatched pred indices)."""
    iou = _iou_matrix(p_box, g_box)
    cands = sorted(((float(iou[i, j]), i, j) for i in range(len(p_box)) for j in range(len(g_box))
                    if iou[i, j] >= iou_thr), reverse=True)
    used_p: set = set()
    used_g: set = set()
    pairs: List[Tuple[int, int]] = []
    for _, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((i, j))
    return pairs, [j for j in range(len(g_box)) if j not in used_g], [i for i in range(len(p_box)) if i not in used_p]


def build_confusion(preds: np.ndarray, gts: np.ndarray, group_of: Sequence[int], n_groups: int,
                    iou_thr: float = 0.5, nms_iou: Optional[float] = None,
                    max_area_frac: Optional[Sequence[float]] = None, image_area: Optional[float] = None) -> np.ndarray:
    """Detection confusion matrix for one image.

    preds: (N,6) [x1,y1,x2,y2,conf,cls]; gts: (M,5) [cls,x1,y1,x2,y2]; group_of maps a class id to a
    group id (identity = native classes, many-to-one = merged classes). Predictions and ground truth are
    matched one-to-one by descending IoU (>= iou_thr), class-agnostic, so a wrong-class hit shows up as an
    off-diagonal cell. Returns an (n_groups+1)^2 int matrix: m[true, predicted]; the extra last row counts
    false positives (no true object), the extra last column counts missed objects. If nms_iou is given,
    same-group predictions are de-duplicated first (what the pipeline's ingestor does after class mapping)."""
    bg = n_groups
    m = np.zeros((n_groups + 1, n_groups + 1), dtype=np.int64)
    p_box = preds[:, :4].astype(float).reshape(-1, 4)
    p_conf = preds[:, 4].astype(float)
    p_grp = np.array([group_of[int(c)] for c in preds[:, 5]], dtype=int)
    if max_area_frac is not None and image_area and len(p_box):  # drop implausibly large boxes (pipeline behaviour)
        areas = (p_box[:, 2] - p_box[:, 0]) * (p_box[:, 3] - p_box[:, 1]) / image_area
        ok = np.array([a <= max_area_frac[g] for a, g in zip(areas, p_grp)], dtype=bool)
        p_box, p_conf, p_grp = p_box[ok], p_conf[ok], p_grp[ok]
    if nms_iou is not None and len(p_box):
        pp = _iou_matrix(p_box, p_box)
        keep: List[int] = []
        for i in np.argsort(-p_conf, kind="stable"):
            if all(not (p_grp[j] == p_grp[i] and pp[i, j] > nms_iou) for j in keep):
                keep.append(int(i))
        p_box, p_conf, p_grp = p_box[keep], p_conf[keep], p_grp[keep]
    g_box = gts[:, 1:5].astype(float).reshape(-1, 4)
    g_grp = np.array([group_of[int(c)] for c in gts[:, 0]], dtype=int)
    pairs, un_g, un_p = match_one_to_one(p_box, g_box, iou_thr)
    for i, j in pairs:
        m[g_grp[j], p_grp[i]] += 1
    for j in un_g:
        m[g_grp[j], bg] += 1
    for i in un_p:
        m[bg, p_grp[i]] += 1
    return m


def read_gts(label: Path, w: int, h: int) -> np.ndarray:
    """YOLO label file -> (M,5) [cls, x1, y1, x2, y2] in pixels for an image of size w x h."""
    rows: List[List[float]] = []
    if label.is_file():
        for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) == 5:
                c, cx, cy, bw, bh = int(float(parts[0])), *(float(v) for v in parts[1:])
                rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=float).reshape(-1, 5)


def summarise_confusion(matrix: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    n = len(labels)
    out: Dict[str, Any] = {}
    for c, name in enumerate(labels):
        tp, true_total, pred_total = int(matrix[c, c]), int(matrix[c, :].sum()), int(matrix[:, c].sum())
        out[name] = {
            "true_objects": true_total, "detected_correct_class": tp,
            "recall": tp / true_total if true_total else None,
            "precision": tp / pred_total if pred_total else None,
            "missed": int(matrix[c, n]), "false_positives": int(matrix[n, c]),
            "detected_as_other_class": {labels[p]: int(matrix[c, p]) for p in range(n) if p != c and matrix[c, p]},
        }
    return out


def confusion_report(weights: Path, data_yaml: Path, split: str = "test", conf: float = 0.25,
                     iou_thr: float = 0.5, imgsz: int = 640,
                     names: Mapping[int, str] = FIGSHARE_CLASS_NAMES,
                     box_limits: Optional[Path] = None) -> Dict[str, Any]:
    """Operating-point diagnostics (confidence `conf`, as in the pipeline): native 4-class confusion and the
    merged project-class confusion (what the quality score actually sees), written to confusion_<split>.json."""
    from .detector import YoloV8Detector
    root = data_yaml.parent
    img_dir, lab_dir = root / "images" / split, root / "labels" / split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not images:
        raise FileNotFoundError(f"No images in {img_dir}")
    det = YoloV8Detector(weights, conf=conf, imgsz=imgsz)
    n_native = len(names)
    native_labels = [names[i] for i in range(n_native)]
    merged_names = [d.name for d in sorted(DefectType, key=int)]
    group_of_merged = [int(FIGSHARE_CLASS_MAP[i]) if FIGSHARE_CLASS_MAP.get(i) is not None else -1 for i in range(n_native)]
    if min(group_of_merged) < 0:
        raise ValueError("Merged analysis needs every class mapped to a DefectType")
    native_lim: Optional[List[float]] = None
    merged_lim: Optional[List[float]] = None
    if box_limits is not None:
        lim = json.loads(Path(box_limits).read_text(encoding="utf-8"))
        native_lim = [float(lim["native"][names[i]]["limit"]) for i in range(n_native)]
        merged_lim = [float(lim["by_defect_type"][n]) for n in merged_names]
    m_native = np.zeros((n_native + 1,) * 2, dtype=np.int64)
    m_merged = np.zeros((len(merged_names) + 1,) * 2, dtype=np.int64)
    for path in images:
        raw, (w, h) = det.detect(path)
        gts = read_gts(lab_dir / f"{path.stem}.txt", w, h)
        m_native += build_confusion(raw, gts, list(range(n_native)), n_native, iou_thr, max_area_frac=native_lim, image_area=float(w * h))
        m_merged += build_confusion(raw, gts, group_of_merged, len(merged_names), iou_thr, nms_iou=0.5,
                                    max_area_frac=merged_lim, image_area=float(w * h))
    report = {
        "weights": str(weights), "split": split, "images": len(images), "confidence": conf, "iou_threshold": iou_thr,
        "box_limits": str(box_limits) if box_limits else None,
        "matrix_layout": "rows = true class (last row = false positives), columns = predicted class (last column = missed)",
        "native": {"labels": native_labels + ["background"], "matrix": m_native.tolist(),
                   "per_class": summarise_confusion(m_native, native_labels)},
        "merged_project_classes": {"labels": merged_names + ["background"], "matrix": m_merged.tolist(),
                                   "per_class": summarise_confusion(m_merged, merged_names),
                                   "note": "cutting+crack merged into MISSING_THREAD with same-class NMS at IoU 0.5, as in the pipeline"},
    }
    atomic_write_json(weights.parent / f"confusion_{split}.json", report)
    return report


def compute_box_limits(data_yaml: Path, splits: Sequence[str] = ("train", "val"), margin: float = 1.25,
                       names: Mapping[int, str] = FIGSHARE_CLASS_NAMES) -> Dict[str, Any]:
    """Per-class distribution of labelled box areas (fraction of the image = w*h, labels are normalised) and
    a data-driven upper limit: limit = min(1, margin * largest labelled box of that class). Only the given
    splits are used (never the test split), so the limit does not leak information from evaluation data."""
    root = data_yaml.parent
    areas: Dict[int, List[float]] = {i: [] for i in names}
    for split in splits:
        folder = root / "labels" / split
        if not folder.is_dir():
            continue
        for lab in sorted(folder.glob("*.txt")):
            for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) == 5 and int(float(parts[0])) in areas:
                    areas[int(float(parts[0]))].append(float(parts[3]) * float(parts[4]))
    native: Dict[str, Any] = {}
    for cid, vals in areas.items():
        if not vals:
            raise ValueError(f"No labelled boxes found for class {names[cid]} in splits {list(splits)}")
        a = np.array(vals)
        native[names[cid]] = {"n": len(vals), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
                              "p99": float(np.percentile(a, 99)), "max": float(a.max()),
                              "limit": float(min(1.0, margin * a.max()))}
    by_type: Dict[str, float] = {}
    for cid, name in names.items():
        mapped = FIGSHARE_CLASS_MAP.get(cid)
        if mapped is not None:
            by_type[mapped.name] = max(by_type.get(mapped.name, 0.0), native[name]["limit"])
    return {"source": str(data_yaml), "splits_used": list(splits), "margin": margin,
            "unit": "box area as a fraction of image area", "native": native, "by_defect_type": by_type}


def _box_stats(items: Sequence[Tuple[float, float, float]]) -> Dict[str, Any]:
    if not items:
        return {"count": 0}
    a = np.array(items, dtype=float)  # columns: short side px, long side px, area % of image
    return {"count": len(items), "median_short_side_px": float(np.median(a[:, 0])),
            "median_long_side_px": float(np.median(a[:, 1])),
            "median_aspect_long_over_short": float(np.median(a[:, 1] / np.maximum(a[:, 0], 1e-9))),
            "median_area_pct_of_image": float(np.median(a[:, 2]))}


def review_images(det: Any, images_dir: Path, labels_dir: Path, target_class: int, out_dir: Path,
                  names: Mapping[int, str] = FIGSHARE_CLASS_NAMES, iou_thr: float = 0.5,
                  max_images: int = 40) -> Dict[str, Any]:
    """For one class, compare found vs missed (or wrong-class) ground-truth boxes and save annotated images
    of the problem cases: GT of the class in green (missed = thick yellow), predictions in red.
    `det` needs .detect(path) -> (raw (N,6), (w,h)). Requires Pillow."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install pillow") from exc
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
    out_dir.mkdir(parents=True, exist_ok=True)
    found: List[Tuple[float, float, float]] = []
    problem: List[Tuple[float, float, float]] = []
    saved: List[str] = []
    wrong_class: Counter[str] = Counter()
    breakdown: Counter[str] = Counter()
    for path in images:
        raw, (w, h) = det.detect(path)
        gts = read_gts(labels_dir / f"{path.stem}.txt", w, h)
        pairs, _, _ = match_one_to_one(raw[:, :4].astype(float).reshape(-1, 4), gts[:, 1:5], iou_thr)
        hit = {j for i, j in pairs if int(raw[i, 5]) == target_class and int(gts[j, 0]) == target_class}
        pred_for = {j: int(raw[i, 5]) for i, j in pairs}
        bad: List[int] = []
        for j in range(len(gts)):
            if int(gts[j, 0]) != target_class:
                continue
            bw, bh = gts[j, 3] - gts[j, 1], gts[j, 4] - gts[j, 2]
            stat = (float(min(bw, bh)), float(max(bw, bh)), 100.0 * float(bw * bh) / (w * h))
            if j in hit:
                found.append(stat)
            else:
                problem.append(stat)
                bad.append(j)
                if j in pred_for:
                    wrong_class[names.get(pred_for[j], str(pred_for[j]))] += 1
                # was there ANY prediction near this box, regardless of class or one-to-one matching?
                best = float(_iou_matrix(raw[:, :4].astype(float).reshape(-1, 4), gts[j:j + 1, 1:5]).max()) if len(raw) else None
                key = ("no_prediction_in_image" if best is None else "best_iou_below_0.1" if best < 0.1
                       else "best_iou_0.1_to_0.3" if best < 0.3 else "best_iou_0.3_to_0.5" if best < iou_thr
                       else "best_iou_at_or_above_threshold(wrong class or box already used)")
                breakdown[key] += 1
        if bad and len(saved) < max_images:
            with Image.open(path) as im:
                canvas = im.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for j in range(len(gts)):
                if int(gts[j, 0]) == target_class:
                    colour, width, tag = ("yellow", 4, "MISSED") if j in bad else ("lime", 2, "GT")
                    draw.rectangle(gts[j, 1:5].tolist(), outline=colour, width=width)
                    draw.text((gts[j, 1] + 3, gts[j, 2] + 3), f"{tag} {names.get(target_class, target_class)}", fill=colour)
            for row in raw:
                draw.rectangle(row[:4].tolist(), outline="red", width=2)
                draw.text((row[0] + 3, row[3] - 12), f"{names.get(int(row[5]), int(row[5]))} {row[4]:.2f}", fill="red")
            target = out_dir / f"{path.stem[:60]}.jpg"
            canvas.save(target)
            saved.append(str(target))
    report = {"class": names.get(target_class, str(target_class)), "iou_threshold": iou_thr,
              "found": _box_stats(found), "missed_or_wrong_class": _box_stats(problem),
              "problem_boxes_detected_as": dict(wrong_class), "problem_breakdown_by_nearest_prediction": dict(breakdown),
              "annotated_images": saved}
    atomic_write_json(out_dir / "review.json", report)
    return report


def format_confusion(report: Dict[str, Any]) -> str:
    out = [f"{report['split']}: {report['images']} images, conf >= {report['confidence']}, IoU >= {report['iou_threshold']}"]
    for key, title in (("native", "NATIVE CLASSES"), ("merged_project_classes", "MERGED (what the quality score sees)")):
        block = report[key]
        labels = block["labels"]
        out.append(f"\n{title}   rows = true, columns = predicted")
        out.append(" " * 16 + "".join(f"{l[:13]:>14}" for l in labels))
        for i, row in enumerate(block["matrix"]):
            name = labels[i] if i < len(labels) - 1 else "(false pos.)"
            out.append(f"{name:<16}" + "".join(f"{v:>14}" for v in row))
        for name, st in block["per_class"].items():
            rec = "n/a" if st["recall"] is None else f"{st['recall']:.3f}"
            pre = "n/a" if st["precision"] is None else f"{st['precision']:.3f}"
            other = f"  confused with {st['detected_as_other_class']}" if st["detected_as_other_class"] else ""
            out.append(f"  {name:<14} true={st['true_objects']:<4} recall={rec}  precision={pre}  missed={st['missed']}  false_pos={st['false_positives']}{other}")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Prepare, train and evaluate the fabric-defect YOLOv8 model")
    sub = p.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("prepare"); a_.add_argument("--src", type=Path, required=True)
    a_.add_argument("--dst", type=Path, required=True); a_.add_argument("--overwrite", action="store_true")
    a_.add_argument("--holdout-date", action="append", default=[], metavar="YYYY-MM-DD",
                    help="use all photos from this date as the test set (repeatable)")
    t_ = sub.add_parser("train"); t_.add_argument("--data", type=Path, required=True)
    t_.add_argument("--weights", default="yolov8n.pt"); t_.add_argument("--epochs", type=int, default=50)
    t_.add_argument("--imgsz", type=int, default=640); t_.add_argument("--batch", type=int, default=16)
    t_.add_argument("--seed", type=int, default=42); t_.add_argument("--device", default=None)
    t_.add_argument("--project", type=Path, default=Path("runs")); t_.add_argument("--name", default="fabric")
    e_ = sub.add_parser("eval"); e_.add_argument("--weights", type=Path, required=True)
    e_.add_argument("--data", type=Path, required=True); e_.add_argument("--split", default="test")
    e_.add_argument("--imgsz", type=int, default=640); e_.add_argument("--device", default=None)
    c_ = sub.add_parser("confusion", help="confusion matrices at the pipeline's confidence (native + merged classes)")
    c_.add_argument("--weights", type=Path, required=True); c_.add_argument("--data", type=Path, required=True)
    c_.add_argument("--split", default="test"); c_.add_argument("--conf", type=float, default=0.25)
    c_.add_argument("--iou", type=float, default=0.5); c_.add_argument("--imgsz", type=int, default=640)
    c_.add_argument("--box-limits", type=Path, default=None, help="apply the oversized-box filter from box_limits.json")
    r_ = sub.add_parser("review", help="save annotated images of missed/wrong-class boxes for one class + size statistics")
    r_.add_argument("--weights", type=Path, required=True); r_.add_argument("--data", type=Path, required=True)
    r_.add_argument("--class-name", default="crack"); r_.add_argument("--split", default="test")
    r_.add_argument("--conf", type=float, default=0.25); r_.add_argument("--iou", type=float, default=0.5)
    r_.add_argument("--imgsz", type=int, default=640); r_.add_argument("--max-images", type=int, default=40)
    r_.add_argument("--out", type=Path, default=None)
    b_ = sub.add_parser("box-limits", help="derive per-class maximum plausible box area from labelled data")
    b_.add_argument("--data", type=Path, required=True); b_.add_argument("--margin", type=float, default=1.25)
    b_.add_argument("--out", type=Path, default=Path("box_limits.json"))
    a = p.parse_args(argv)
    if a.cmd == "box-limits":
        rep = compute_box_limits(a.data, margin=a.margin)
        atomic_write_json(a.out, rep)
        for name, st in rep["native"].items():
            print(f"{name:<10} n={st['n']:<4} p50={st['p50']*100:5.2f}%  p90={st['p90']*100:5.2f}%  p99={st['p99']*100:5.2f}%  max={st['max']*100:5.2f}%  limit={st['limit']*100:5.2f}%")
        print(f"by defect type: { {k: round(v*100, 2) for k, v in rep['by_defect_type'].items()} } (% of image)  -> {a.out}")
        return 0
    if a.cmd == "review":
        from .detector import YoloV8Detector
        inv = {v: k for k, v in FIGSHARE_CLASS_NAMES.items()}
        if a.class_name not in inv:
            raise SystemExit(f"--class-name must be one of {sorted(inv)}")
        root = a.data.parent
        rep = review_images(YoloV8Detector(a.weights, conf=a.conf, imgsz=a.imgsz), root / "images" / a.split,
                            root / "labels" / a.split, inv[a.class_name], a.out or Path(f"review_{a.class_name}"),
                            iou_thr=a.iou, max_images=a.max_images)
        print(json.dumps({k: rep[k] for k in ("class", "iou_threshold", "found", "missed_or_wrong_class", "problem_boxes_detected_as", "problem_breakdown_by_nearest_prediction")}, indent=2))
        print(f"{len(rep['annotated_images'])} annotated images saved; open the folder and look at them.")
        return 0
    if a.cmd == "confusion":
        print(format_confusion(confusion_report(a.weights, a.data, a.split, a.conf, a.iou, a.imgsz, box_limits=a.box_limits)))
        return 0
    if a.cmd == "prepare":
        rep = prepare_dataset(a.src, a.dst, overwrite=a.overwrite, holdout_dates=a.holdout_date)
        print(json.dumps({k: rep[k] for k in ("images_kept", "dropped_duplicates", "holdout_dates", "per_split", "box_reconciliation")}, indent=2))
    elif a.cmd == "train":
        print(train_model(a.data, a.weights, a.epochs, a.imgsz, a.batch, a.seed, a.device, a.project, a.name))
    else:
        print(json.dumps(evaluate_model(a.weights, a.data, a.split, a.imgsz, a.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

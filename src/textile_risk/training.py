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

from .common import atomic_write_json
from .datacheck import IMAGE_EXT, NON_LABEL_STEMS, SPLIT_NAMES, inspect_yolo_dataset, photo_datetime
from .quality import FIGSHARE_CLASS_NAMES

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
    a = p.parse_args(argv)
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

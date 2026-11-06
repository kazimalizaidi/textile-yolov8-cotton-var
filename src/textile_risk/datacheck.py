"""Input checkers: price CSV files and YOLO-format defect datasets.

    textile-risk-check prices data/cotton_polyester_monthly_fred.csv
    textile-risk-check defects data/defects/figshare_fd
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .risk import MIN_OBSERVATIONS, PERIOD_UNIT, MarketDataProvider

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
OTHER_ANNOTATION_EXT = {".xml", ".json", ".csv"}
NON_LABEL_STEMS = {"classes", "obj", "readme", "notes", "train", "val", "valid", "test", "labels"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}
PERIODS_PER_YEAR = {"daily": 252.0, "weekly": 52.0, "monthly": 12.0}
TIMESTAMP_RE = re.compile(r"photo_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})")


def photo_datetime(name: str) -> Optional[datetime]:
    """'photo_2024-02-11_14-43-15 (2).jpg' or '..._jpg.rf.<hash>.jpg' -> datetime(2024, 2, 11, 14, 43, 15)."""
    m = TIMESTAMP_RE.search(name)
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()))
    except ValueError:
        return None


# ----------------------------------------------------------------------------- prices
def inspect_prices(path: Path, frequency: str = "auto") -> Dict[str, Any]:
    """Load a price CSV exactly as the pipeline would and report what it will see."""
    market = MarketDataProvider.from_csv(path, frequency)
    rets = market.log_returns
    n = len(rets)
    required = MIN_OBSERVATIONS[market.frequency]
    columns: Dict[str, Any] = {}
    for col in rets.columns:
        r = rets[col]
        sigma = float(r.std(ddof=1))
        z = (r - r.mean()) / sigma if sigma > 0 else r * 0.0
        columns[col] = {
            "source_column": market.source_columns.get(col, col),
            "first": float(market.prices[col].iloc[0]), "last": float(market.prices[col].iloc[-1]),
            "min": float(market.prices[col].min()), "max": float(market.prices[col].max()),
            "period_vol_pct": 100.0 * sigma,
            "annualised_vol_pct": 100.0 * sigma * np.sqrt(PERIODS_PER_YEAR[market.frequency]),
            "zero_return_share_pct": 100.0 * float((r == 0).mean()),
            "largest_move_pct": 100.0 * float(np.expm1(r.loc[r.abs().idxmax()])),
            "largest_move_date": str(r.abs().idxmax().date()),
            "n_beyond_5_sigma": int((z.abs() > 5).sum()),
        }
    gaps = np.diff(market.prices.index.values).astype("timedelta64[D]").astype(float)
    return {"rows": len(market.prices), "returns": n, "start": str(market.prices.index[0].date()),
            "end": str(market.prices.index[-1].date()), "frequency": market.frequency,
            "required_observations": required, "meets_minimum": n >= required,
            "tail_observations_99": n * 0.01, "median_gap_days": float(np.median(gaps)),
            "max_gap_days": float(gaps.max()),
            "correlation": float(rets.corr().iloc[0, 1]), "columns": columns,
            "period_unit": PERIOD_UNIT[market.frequency]}


def format_prices_report(rep: Dict[str, Any]) -> str:
    lines = [f"rows={rep['rows']}  returns={rep['returns']}  {rep['start']} -> {rep['end']}",
             f"frequency={rep['frequency']}  median gap={rep['median_gap_days']:.0f}d  max gap={rep['max_gap_days']:.0f}d",
             f"correlation of returns = {rep['correlation']:.3f}"]
    for col, c in rep["columns"].items():
        lines.append(f"[{col}] <- file column '{c['source_column']}'  last={c['last']:.4f}  range {c['min']:.4f}..{c['max']:.4f}")
        lines.append(f"    vol/{rep['period_unit']}={c['period_vol_pct']:.3f}%  annualised={c['annualised_vol_pct']:.2f}%  "
                     f"zero-return share={c['zero_return_share_pct']:.1f}%  largest move={c['largest_move_pct']:+.2f}% "
                     f"({c['largest_move_date']})  >5sigma moves={c['n_beyond_5_sigma']}")
        if c["zero_return_share_pct"] > 10.0:
            lines.append("    WARNING: >10% of periods unchanged - series is sticky/smoothed, volatility is understated")
    ok = "OK" if rep["meets_minimum"] else "BELOW MINIMUM (run is rejected unless --min-observations is lowered)"
    lines.append(f"observations {rep['returns']} vs required {rep['required_observations']}: {ok}")
    lines.append(f"99% tail observations = {rep['tail_observations_99']:.2f}"
                 + ("  WARNING: <10, 99% VaR/ES statistically unreliable" if rep["tail_observations_99"] < 10 else ""))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- YOLO
@dataclass
class YoloReport:
    root: str
    n_images: int = 0
    n_label_files: int = 0
    n_boxes: int = 0
    images_without_label: List[str] = field(default_factory=list)
    unlabeled_unsplit: List[str] = field(default_factory=list)
    labels_without_image: List[str] = field(default_factory=list)
    empty_labels: int = 0
    bad_lines: List[str] = field(default_factory=list)
    segmentation_lines: int = 0
    class_counts: Dict[int, int] = field(default_factory=dict)
    class_names: Dict[int, str] = field(default_factory=dict)
    splits: Dict[str, int] = field(default_factory=dict)
    duplicate_stems: int = 0
    other_annotation_files: Dict[str, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not (self.bad_lines or self.images_without_label or self.labels_without_image or self.n_images == 0)


def _read_class_names(root: Path) -> Dict[int, str]:
    for p in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"^names\s*:\s*\[(.*?)\]", text, re.M | re.S)
        if m:
            items = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
            return dict(enumerate(items))
        pairs = re.findall(r"^\s+(\d+)\s*:\s*['\"]?([^'\"\n]+)['\"]?\s*$", text, re.M)
        if pairs:
            return {int(k): v.strip() for k, v in pairs}
        dash = re.search(r"^names\s*:\s*\n((?:\s*-\s*.+\n?)+)", text, re.M)
        if dash:
            return dict(enumerate(s.strip().strip("'\"") for s in re.findall(r"-\s*(.+)", dash.group(1))))
    for p in list(root.rglob("classes.txt")) + list(root.rglob("*.names")):
        return dict(enumerate(l.strip() for l in p.read_text(errors="ignore").splitlines() if l.strip()))
    return {}


def inspect_yolo_dataset(root: Path) -> YoloReport:
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    rep = YoloReport(root=str(root))
    images: Dict[str, Path] = {}
    labels: Dict[str, Path] = {}
    image_split: Dict[str, str] = {}
    other: Counter[str] = Counter()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXT:
            if p.stem in images:
                rep.duplicate_stems += 1
            images[p.stem] = p
            split = next((q.lower() for q in p.parts if q.lower() in SPLIT_NAMES), "unsplit")
            rep.splits[split] = rep.splits.get(split, 0) + 1
            image_split[p.stem] = split
        elif ext == ".txt" and p.stem.lower() not in NON_LABEL_STEMS:
            labels[p.stem] = p
        elif ext in OTHER_ANNOTATION_EXT:
            other[ext] += 1
    rep.n_images, rep.n_label_files = len(images) + rep.duplicate_stems, len(labels)
    rep.other_annotation_files = dict(other)
    rep.class_names = _read_class_names(root)
    missing = [s for s in images if s not in labels]
    if any(k != "unsplit" for k in rep.splits):  # dataset has train/val/test folders
        rep.images_without_label = sorted(s for s in missing if image_split[s] != "unsplit")
        rep.unlabeled_unsplit = sorted(s for s in missing if image_split[s] == "unsplit")
    else:
        rep.images_without_label = sorted(missing)
    rep.labels_without_image = sorted(s for s in labels if s not in images)

    counts: Counter[int] = Counter()
    for stem, p in labels.items():
        lines = [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        if not lines:
            rep.empty_labels += 1
        for i, line in enumerate(lines, 1):
            parts = line.split()
            where = f"{p.name}:{i}"
            try:
                cls_f = float(parts[0])
                if cls_f != int(cls_f) or int(cls_f) < 0:
                    raise ValueError("class id must be a non-negative integer")
                vals = [float(x) for x in parts[1:]]
            except (ValueError, IndexError):
                rep.bad_lines.append(f"{where}: unparseable '{line[:40]}'")
                continue
            if len(vals) == 4:
                x, y, w, h = vals
                if not all(0.0 <= v <= 1.0 for v in vals) or w <= 0 or h <= 0:
                    rep.bad_lines.append(f"{where}: coordinates out of [0,1] or zero-size '{line[:40]}'")
                    continue
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                rep.segmentation_lines += 1  # polygon format: valid YOLO-seg, not a box
            else:
                rep.bad_lines.append(f"{where}: expected 5 fields, got {len(parts)}")
                continue
            counts[int(cls_f)] += 1
            rep.n_boxes += 1
    rep.class_counts = dict(sorted(counts.items()))
    return rep


def format_yolo_report(rep: YoloReport) -> str:
    out = [f"root: {rep.root}",
           f"images={rep.n_images}  label files={rep.n_label_files}  boxes={rep.n_boxes}  empty labels={rep.empty_labels}",
           f"splits: {rep.splits}"]
    if rep.class_names:
        out.append(f"class names: {rep.class_names}")
    else:
        out.append("class names: not found (no data.yaml / classes.txt) - check the dataset docs")
    out.append("class counts (id: boxes): " + ", ".join(
        f"{k}({rep.class_names.get(k, '?')}): {v}" for k, v in rep.class_counts.items()))
    if rep.segmentation_lines:
        out.append(f"NOTE: {rep.segmentation_lines} polygon (segmentation) lines - not plain boxes")
    if rep.other_annotation_files:
        out.append(f"other annotation files present: {rep.other_annotation_files} (dataset may not be YOLO-only)")
    if rep.duplicate_stems:
        out.append(f"WARNING: {rep.duplicate_stems} images share a filename stem; pairing by name is ambiguous")
    if rep.unlabeled_unsplit:
        out.append(f"NOTE: {len(rep.unlabeled_unsplit)} unlabeled images sit outside the train/valid/test folders "
                   f"(likely raw originals; they are ignored for training)")
    for title, items in (("images without label", rep.images_without_label),
                         ("labels without image", rep.labels_without_image), ("bad label lines", rep.bad_lines)):
        if items:
            out.append(f"{title}: {len(items)}  e.g. {items[:5]}")
    out.append("RESULT: " + ("clean" if rep.clean else "issues found (see above)"))
    return "\n".join(out)


def make_class_previews(root: Path, out_dir: Path, per_class: int = 8, tile: int = 160,
                        columns: int = 4) -> Dict[int, Path]:
    """Save one contact sheet per class id made of cropped boxes, so a human can see what each
    numeric class really is. Requires Pillow (pip install pillow)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for previews: pip install pillow") from exc

    images: Dict[str, Path] = {}
    labels: Dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            images[p.stem] = p
        elif p.is_file() and p.suffix.lower() == ".txt" and p.stem.lower() not in NON_LABEL_STEMS:
            labels[p.stem] = p
    boxes: Dict[int, List[tuple]] = {}
    for stem, lab in sorted(labels.items()):
        if stem not in images:
            continue
        for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls, x, y, w, h = int(float(parts[0])), *(float(v) for v in parts[1:])
            except ValueError:
                continue
            boxes.setdefault(cls, []).append((images[stem], x, y, w, h))

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[int, Path] = {}
    for cls, items in sorted(boxes.items()):
        picks = [items[int(i)] for i in np.linspace(0, len(items) - 1, min(per_class, len(items)))]
        rows = -(-len(picks) // columns)
        sheet = Image.new("RGB", (columns * tile, rows * tile), (128, 128, 128))
        for k, (img_path, x, y, w, h) in enumerate(picks):
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                mx, my = 0.15 * w * W, 0.15 * h * H  # context margin around the box
                box = (max(0, int((x - w / 2) * W - mx)), max(0, int((y - h / 2) * H - my)),
                       min(W, int((x + w / 2) * W + mx)), min(H, int((y + h / 2) * H + my)))
                crop = im.crop(box)
                crop.thumbnail((tile, tile))
                sheet.paste(crop, ((k % columns) * tile, (k // columns) * tile))
        saved[cls] = out_dir / f"class_{cls}.png"
        sheet.save(saved[cls])
    return saved


def _dhash_variants(im: Any, size: int = 8) -> List[int]:
    """64-bit difference hashes of the image and its flips/rotations (catches simple augmentation)."""
    from PIL import Image
    variants = [im, im.transpose(Image.FLIP_LEFT_RIGHT), im.transpose(Image.FLIP_TOP_BOTTOM),
                im.transpose(Image.ROTATE_90), im.transpose(Image.ROTATE_180), im.transpose(Image.ROTATE_270)]
    hashes: List[int] = []
    for v in variants:
        g = np.asarray(v.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.float32)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        hashes.append(int("".join("1" if b else "0" for b in bits), 2))
    return hashes


def find_cross_split_near_duplicates(root: Path, threshold: int = 6, sheet: Optional[Path] = None,
                                     max_pairs: int = 12) -> Dict[str, Any]:
    """For a prepared dataset (images/train|val|test), find val/test images whose perceptual hash is
    within `threshold` bits (of 64) of a train image, or a test image near a val image, allowing
    flips/rotations. Near-duplicates across splits inflate validation/test scores."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install pillow") from exc
    order = ["train", "val", "test"]
    files: Dict[str, List[Path]] = {s: sorted(p for p in (root / "images" / s).glob("*") if p.suffix.lower() in IMAGE_EXT)
                                    for s in order if (root / "images" / s).is_dir()}
    if "train" not in files:
        raise FileNotFoundError(f"{root}/images/train not found (run textile-risk-train prepare first)")
    base: Dict[str, np.ndarray] = {}
    variants: Dict[str, List[List[int]]] = {}
    for split, paths in files.items():
        hs = []
        for path in paths:
            with Image.open(path) as im:
                hs.append(_dhash_variants(im.convert("RGB")))
        variants[split] = hs
        base[split] = np.array([h[0] for h in hs], dtype=np.uint64)

    def popcount(x: np.ndarray) -> np.ndarray:
        return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)

    result: Dict[str, Any] = {"threshold_bits": threshold, "splits": {}, "pairs": []}
    pairs: List[tuple] = []
    for i, split in enumerate(order):
        if split not in files or i == 0:
            continue
        ref_splits = [s for s in order[:i] if s in files]
        ref_paths = [(s, p) for s in ref_splits for p in files[s]]
        ref = np.concatenate([base[s] for s in ref_splits])
        hist = {"0-2": 0, "3-6": 0, "7-12": 0, "13+": 0}
        near = 0
        for path, hs in zip(files[split], variants[split]):
            best = (99, -1)
            for v in hs:
                d = popcount(ref ^ np.uint64(v))
                j = int(d.argmin())
                best = min(best, (int(d[j]), j))
            dist, j = best
            hist["0-2" if dist <= 2 else "3-6" if dist <= 6 else "7-12" if dist <= 12 else "13+"] += 1
            if dist <= threshold:
                near += 1
                pairs.append((dist, split, path, ref_paths[j][0], ref_paths[j][1]))
        result["splits"][split] = {"images": len(files[split]), "near_duplicates": near,
                                   "share_pct": 100.0 * near / max(len(files[split]), 1), "min_distance_histogram": hist}
    pairs.sort(key=lambda t: t[0])
    result["pairs"] = [{"distance": d, "query": f"{s}/{q.name}", "match": f"{ms}/{m.name}"} for d, s, q, ms, m in pairs[:max_pairs]]
    if sheet is not None and pairs:
        tile = 200
        canvas = Image.new("RGB", (2 * tile, tile * min(len(pairs), max_pairs)), (128, 128, 128))
        for r, (_, _, q, _, m) in enumerate(pairs[:max_pairs]):
            for c, path in enumerate((q, m)):
                with Image.open(path) as im:
                    t = im.convert("RGB")
                    t.thumbnail((tile, tile))
                    canvas.paste(t, (c * tile, r * tile))
        sheet.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(sheet)
        result["sheet"] = str(sheet)
    return result


def inspect_sessions(root: Path, windows: Sequence[int] = (10, 60, 600)) -> Dict[str, Any]:
    """Shooting-session structure of a prepared dataset: photos per date in each split, and how close in
    time each val/test photo is to its nearest train photo (a proxy for the 'neighbouring shots' effect)."""
    stamps: Dict[str, List[datetime]] = {}
    unparsed: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        folder = root / "images" / split
        if not folder.is_dir():
            continue
        parsed = [photo_datetime(p.name) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXT]
        stamps[split] = [x for x in parsed if x is not None]
        unparsed[split] = sum(1 for x in parsed if x is None)
    if "train" not in stamps or not stamps["train"]:
        raise FileNotFoundError(f"No timestamped train images under {root}/images/train")
    tr = np.array([x.timestamp() for x in stamps["train"]])
    rep: Dict[str, Any] = {"splits": {}, "windows_s": list(windows)}
    for split, items in stamps.items():
        dates = Counter(x.date().isoformat() for x in items)
        entry: Dict[str, Any] = {"timestamped": len(items), "without_timestamp": unparsed[split],
                                 "photos_by_date": dict(sorted(dates.items()))}
        q = np.array([x.timestamp() for x in items])
        if split == "train":
            gaps = np.array([np.min(np.abs(np.delete(tr, i) - t)) for i, t in enumerate(tr)]) if len(tr) > 1 else np.array([])
            entry["role"] = "nearest OTHER train photo (baseline)"
        else:
            gaps = np.array([np.min(np.abs(tr - t)) for t in q]) if len(q) else np.array([])
            entry["role"] = "nearest train photo"
        if gaps.size:
            entry["median_gap_s"] = float(np.median(gaps))
            entry["within_window"] = {f"{w}s": int((gaps <= w).sum()) for w in windows}
            entry["share_within_window_pct"] = {f"{w}s": 100.0 * float((gaps <= w).mean()) for w in windows}
        rep["splits"][split] = entry
    return rep


def format_sessions_report(rep: Dict[str, Any]) -> str:
    out: List[str] = []
    for split, e in rep["splits"].items():
        out.append(f"[{split}] {e['timestamped']} photos (+{e['without_timestamp']} without timestamp)  by date: {e['photos_by_date']}")
        if "median_gap_s" in e:
            shares = ", ".join(f"<= {k}: {e['within_window'][k]} ({e['share_within_window_pct'][k]:.0f}%)" for k in e["within_window"])
            out.append(f"    {e['role']}: median gap {e['median_gap_s']:.0f}s | {shares}")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Check price CSVs and YOLO defect datasets before using them")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prices", help="validate a price CSV exactly as the pipeline loads it")
    pp.add_argument("csv", type=Path)
    pp.add_argument("--frequency", choices=["auto", "daily", "weekly", "monthly"], default="auto")
    pd_ = sub.add_parser("defects", help="validate a YOLO-format defect dataset folder")
    pd_.add_argument("folder", type=Path)
    pv = sub.add_parser("preview", help="save per-class contact sheets of cropped defects (needs Pillow)")
    pv.add_argument("folder", type=Path)
    pv.add_argument("--out", type=Path, default=Path("class_previews"))
    pv.add_argument("--per-class", type=int, default=8)
    lk = sub.add_parser("leakage", help="find near-duplicate images across train/val/test (needs Pillow)")
    lk.add_argument("prepared", type=Path, help="folder produced by textile-risk-train prepare")
    lk.add_argument("--threshold", type=int, default=6, help="max Hamming distance (of 64 bits) to call a near-duplicate")
    lk.add_argument("--sheet", type=Path, default=None, help="save a side-by-side sheet of the closest pairs (left=query, right=train/val match)")
    se = sub.add_parser("sessions", help="photos per shooting date and time gaps between test/val and train photos")
    se.add_argument("prepared", type=Path, help="folder produced by textile-risk-train prepare")
    a = p.parse_args(argv)
    if a.cmd == "sessions":
        try:
            print(format_sessions_report(inspect_sessions(a.prepared)))
        except FileNotFoundError as exc:
            print(f"FAIL: {exc}")
            return 2
        return 0
    if a.cmd == "leakage":
        try:
            rep = find_cross_split_near_duplicates(a.prepared, a.threshold, a.sheet)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"FAIL: {exc}")
            return 2
        for split, r in rep["splits"].items():
            print(f"{split}: {r['near_duplicates']}/{r['images']} images ({r['share_pct']:.1f}%) within {rep['threshold_bits']} bits of an earlier split; "
                  f"min-distance histogram {r['min_distance_histogram']}")
        for pr in rep["pairs"][:10]:
            print(f"  d={pr['distance']:2d}  {pr['query']}  ~  {pr['match']}")
        if "sheet" in rep:
            print(f"sheet: {rep['sheet']}")
        return 0 if all(r["near_duplicates"] == 0 for r in rep["splits"].values()) else 1
    if a.cmd == "preview":
        try:
            saved = make_class_previews(a.folder, a.out, a.per_class)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"FAIL: {exc}")
            return 2
        for cls, path in saved.items():
            print(f"class {cls}: {path}")
        return 0
    if a.cmd == "prices":
        try:
            print(format_prices_report(inspect_prices(a.csv, a.frequency)))
        except (ValueError, FileNotFoundError) as exc:
            print(f"FAIL: {exc}")
            return 2
        return 0
    try:
        rep = inspect_yolo_dataset(a.folder)
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}")
        return 2
    print(format_yolo_report(rep))
    return 0 if rep.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())

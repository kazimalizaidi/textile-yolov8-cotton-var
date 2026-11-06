from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from textile_risk.datacheck import inspect_prices, inspect_yolo_dataset
from textile_risk.risk import MarketDataProvider, detect_frequency

DATA = Path(__file__).resolve().parents[1] / "data"


def _make_yolo(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for stem in ("a", "b", "d"):
        (root / "images" / "train" / f"{stem}.jpg").write_bytes(b"x")
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    (root / "labels" / "train" / "c.txt").write_text("0 0.5 0.5 0.2 0.2\n")  # orphan (no c.jpg)
    (root / "labels" / "train" / "d.txt").write_text("0 1.5 0.5 0.2 0.2\nx 0.1 0.1 0.1 0.1\n2 0.1 0.1 0.1\n")
    (root / "data.yaml").write_text("names:\n  0: hole\n  1: oil\n  2: thread\n")


def test_yolo_inspector_counts_and_flags_problems(tmp_path):
    _make_yolo(tmp_path)
    rep = inspect_yolo_dataset(tmp_path)
    assert rep.n_images == 3 and rep.n_label_files == 3 and rep.n_boxes == 3  # a:2 + c:1 valid
    assert rep.class_counts == {0: 2, 1: 1}
    assert rep.class_names == {0: "hole", 1: "oil", 2: "thread"}
    assert rep.images_without_label == ["b"] and rep.labels_without_image == ["c"]
    assert len(rep.bad_lines) == 3 and not rep.clean
    assert rep.splits == {"train": 3}


def test_yolo_inspector_clean_dataset(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.png").write_bytes(b"x")
    (tmp_path / "img" / "a.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    rep = inspect_yolo_dataset(tmp_path)
    assert rep.clean and rep.n_boxes == 1
    with pytest.raises(FileNotFoundError):
        inspect_yolo_dataset(tmp_path / "nope")


def test_detect_frequency():
    assert detect_frequency(pd.bdate_range("2024-01-01", periods=50)) == "daily"
    assert detect_frequency(pd.date_range("2024-01-01", periods=50, freq="W")) == "weekly"
    assert detect_frequency(pd.date_range("2024-01-01", periods=50, freq="MS")) == "monthly"


def test_real_monthly_file_is_loaded_with_aliases_and_detected():
    md = MarketDataProvider.from_csv(DATA / "cotton_polyester_monthly_fred.csv")
    assert md.frequency == "monthly" and len(md.prices) == 184
    assert md.source_columns["cotton_futures"] == "cotton_global_usc_per_lb"
    assert md.source_columns["polyester_spot"] == "polyester_ppi_index"


def test_real_data_known_values():
    """Anchor values copied from the FRED tables (cotton Sep-2026 and polyester Sep-2026, Jun-2011 base)."""
    df = pd.read_csv(DATA / "cotton_polyester_monthly_fred.csv", parse_dates=["date"]).set_index("date")
    assert df.loc["2011-06-01", "polyester_ppi_index"] == 100.0
    assert df.loc["2026-09-01", "polyester_ppi_index"] == pytest.approx(141.426)
    assert df.loc["2025-12-01", "cotton_global_usc_per_lb"] == pytest.approx(74.065909, abs=1e-5)
    # FRED's own annual means for the cotton series
    assert df.loc["2025", "cotton_global_usc_per_lb"].mean() == pytest.approx(77.43511, abs=1e-5)
    assert df.loc["2022", "cotton_global_usc_per_lb"].mean() == pytest.approx(130.13901, abs=1e-5)


def test_prices_inspector_reports_real_file_limits():
    rep = inspect_prices(DATA / "cotton_polyester_monthly_fred.csv")
    assert rep["returns"] == 183 and rep["frequency"] == "monthly" and rep["meets_minimum"]
    assert rep["tail_observations_99"] == pytest.approx(1.83)
    assert rep["columns"]["polyester_spot"]["zero_return_share_pct"] > 10  # sticky PPI
    assert rep["columns"]["cotton_futures"]["annualised_vol_pct"] == pytest.approx(17.18, abs=0.05)


def test_pipeline_var_on_real_data_matches_independent_pandas(tmp_path):
    """Recompute 1-month VaR/ES for the real file with plain pandas, bypassing the engine classes."""
    import json

    from textile_risk.orchestrator import PlatformConfig, TextileRiskPlatform
    cfg = PlatformConfig(output_dir=tmp_path, data_source="csv",
                         prices_csv=DATA / "cotton_polyester_monthly_fred.csv")
    TextileRiskPlatform(cfg).run_risk_engine()
    got = json.loads((tmp_path / "hedge_config.json").read_text())["risk_metrics"]["baseline_historical"]

    df = pd.read_csv(DATA / "cotton_polyester_monthly_fred.csv", parse_dates=["date"]).set_index("date")
    r = np.log(df / df.shift(1)).dropna()
    pnl = 2_000_000 * (np.exp(r["cotton_global_usc_per_lb"]) - 1) + 1_500_000 * (np.exp(r["polyester_ppi_index"]) - 1)
    q95, q99 = pnl.quantile(0.05), pnl.quantile(0.01)
    assert got["var_95"] == pytest.approx(-q95, abs=0.01) and got["var_99"] == pytest.approx(-q99, abs=0.01)
    assert got["es_95"] == pytest.approx(-pnl[pnl <= q95].mean(), abs=0.01)
    assert got["es_99"] == pytest.approx(-pnl[pnl <= q99].mean(), abs=0.01)
    assert got["n_scenarios"] == 183


def test_alias_missing_column_gives_clear_error(tmp_path):
    f = tmp_path / "x.csv"
    pd.DataFrame({"date": ["2024-01-01", "2024-02-01", "2024-03-01"], "cotton_futures": [70, 71, 72]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="polyester"):
        MarketDataProvider.from_csv(f)


def test_unlabeled_raw_images_outside_splits_are_not_an_error(tmp_path):
    for split in ("train", "valid"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        (tmp_path / split / "images" / f"{split}1.jpg").write_bytes(b"x")
        (tmp_path / split / "labels" / f"{split}1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "orig1.jpg").write_bytes(b"x")  # raw original, no label, outside splits
    rep = inspect_yolo_dataset(tmp_path)
    assert rep.clean and rep.unlabeled_unsplit == ["orig1"] and rep.images_without_label == []
    (tmp_path / "train" / "images" / "extra.jpg").write_bytes(b"x")  # unlabeled inside a split IS an error
    assert not inspect_yolo_dataset(tmp_path).clean


def test_class_previews_written(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    from textile_risk.datacheck import make_class_previews
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for i in range(5):
        Image.new("RGB", (200, 100), (i * 40, 90, 200)).save(tmp_path / "images" / f"im{i}.jpg")
        (tmp_path / "labels" / f"im{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.3 0.4\n")
    saved = make_class_previews(tmp_path, tmp_path / "out", per_class=3, tile=64, columns=2)
    assert set(saved) == {0, 1} and all(p.exists() for p in saved.values())
    with Image.open(saved[0]) as im:
        assert im.size == (128, 128)  # 3 crops -> 2 columns x 2 rows of 64 px


def _noise_img(seed: int, size=(96, 96)):
    from PIL import Image
    rng = np.random.default_rng(seed)
    small = rng.random((12, 12, 3))
    return Image.fromarray((small * 255).astype("uint8")).resize(size, Image.BICUBIC)


def test_leakage_detects_flipped_and_brightened_copies_but_not_unrelated(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image, ImageEnhance

    from textile_risk.datacheck import find_cross_split_near_duplicates
    for s in ("train", "val", "test"):
        (tmp_path / "images" / s).mkdir(parents=True)
    a, b = _noise_img(1), _noise_img(2)
    a.save(tmp_path / "images" / "train" / "a.png")
    b.save(tmp_path / "images" / "train" / "b.png")
    a.transpose(Image.FLIP_LEFT_RIGHT).save(tmp_path / "images" / "val" / "a_flip.png")        # leak (flip)
    ImageEnhance.Brightness(b).enhance(1.2).save(tmp_path / "images" / "test" / "b_bright.png")  # leak (brightness)
    _noise_img(99).save(tmp_path / "images" / "test" / "unrelated.png")                         # clean
    rep = find_cross_split_near_duplicates(tmp_path, threshold=6, sheet=tmp_path / "sheet.png")
    assert rep["splits"]["val"]["near_duplicates"] == 1
    assert rep["splits"]["test"]["near_duplicates"] == 1 and rep["splits"]["test"]["images"] == 2
    assert {p["query"] for p in rep["pairs"]} == {"val/a_flip.png", "test/b_bright.png"}
    assert (tmp_path / "sheet.png").exists()

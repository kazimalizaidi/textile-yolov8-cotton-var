"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .orchestrator import PlatformConfig, TextileRiskPlatform


def parse_args(argv: Optional[Sequence[str]] = None) -> PlatformConfig:
    p = argparse.ArgumentParser(description="Textile quality / event hedging / VaR backend")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--source", dest="data_source", default="synthetic", choices=["synthetic", "csv", "yahoo"])
    p.add_argument("--prices-csv", type=Path, help="CSV: date, cotton_futures (USc/lb), polyester_spot (USD/kg)")
    p.add_argument("--polyester-csv", type=Path, help="CSV: date, polyester_spot (used with --source yahoo)")
    p.add_argument("--cotton-notional", type=float, default=2_000_000.0)
    p.add_argument("--polyester-notional", type=float, default=1_500_000.0)
    p.add_argument("--horizon-days", type=int, default=1)
    p.add_argument("--tariff-rate", type=float, default=0.20)
    p.add_argument("--tariff-direction", type=int, choices=[-1, 1], default=-1)
    p.add_argument("--hedge-budget", type=float, default=2_000_000.0)
    p.add_argument("--frequency", choices=["auto", "daily", "weekly", "monthly"], default="auto",
                   help="price-file frequency (auto = detect from dates)")
    p.add_argument("--min-observations", type=int, default=None,
                   help="override minimum return observations (default: 250 daily / 104 weekly / 120 monthly)")
    p.add_argument("--detector", choices=["simulator", "yolo"], default="simulator")
    p.add_argument("--weights", type=Path, help="trained YOLOv8 weights (best.pt) for --detector yolo")
    p.add_argument("--images-dir", type=Path, help="folder of fabric images (one image = one frame)")
    p.add_argument("--frame-width-m", type=float, help="physical fabric width visible in one image (metres)")
    p.add_argument("--frame-length-m", type=float, help="physical fabric length visible in one image (metres)")
    p.add_argument("--class-map", default="figshare",
                   help="'figshare' or e.g. '0:oil_stain,1:missing_thread,2:ignore,3:hole'")
    p.add_argument("--box-limits", type=Path, help="box_limits.json: reject implausibly large boxes (see textile-risk-train box-limits)")
    p.add_argument("--detect-conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args(argv)
    return PlatformConfig(output_dir=a.output_dir, seed=a.seed, workers=a.workers, log_level=a.log_level,
                          data_source=a.data_source, prices_csv=a.prices_csv, polyester_csv=a.polyester_csv,
                          cotton_notional_usd=a.cotton_notional, polyester_notional_usd=a.polyester_notional,
                          horizon_days=a.horizon_days, tariff_rate=a.tariff_rate,
                          tariff_direction=a.tariff_direction, hedge_budget_usd=a.hedge_budget,
                          frequency=a.frequency, min_observations=a.min_observations,
                          detector=a.detector, weights=a.weights, images_dir=a.images_dir,
                          frame_width_m=a.frame_width_m, frame_length_m=a.frame_length_m,
                          class_map=a.class_map, detect_conf=a.detect_conf, imgsz=a.imgsz,
                          box_limits=a.box_limits)


def main(argv: Optional[Sequence[str]] = None) -> None:
    TextileRiskPlatform(parse_args(argv)).run()

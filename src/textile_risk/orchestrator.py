"""Orchestration: wires the three modules into one runnable pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import atomic_write_json, configure_logging, utc_now_iso
from .events import (ConsecutiveSevereDefectDetector, Event, EventType, HedgeConfig,
                     InMemoryEventBroker, MockSupplierApiClient, Supplier,
                     SupplierHedgeOrchestrator)
from .datacheck import IMAGE_EXT
from .quality import (BatchQualityScorer, BatchReport, Detection, DetectionIngestor,
                      FrameGeometry, QualityConfig, YoloV8Simulator, load_box_limits, parse_class_map)
from .risk import (COTTON_COL, MIN_OBSERVATIONS, POLYESTER_COL, HedgeConfigWriter, HedgeConstraints,
                   HedgeOptimizer, HistoricalVaREngine, MarketData, MarketDataProvider,
                   TariffShockSpec, TariffStressTester)

LOGGER = logging.getLogger("textile.platform")


@dataclass(frozen=True)
class BatchSpec:
    batch_id: str
    n_frames: int  # metres inspected (1 frame = 1 m)
    defect_rate: float  # mean defects per frame
    severe_burst: Optional[Tuple[int, int]] = None  # frames [a, b) with a forced severe hole each


DEFAULT_BATCHES: Tuple[BatchSpec, ...] = (
    BatchSpec("B-1001", 400, 0.12),
    BatchSpec("B-1002", 400, 0.35),
    BatchSpec("B-1003", 400, 0.80),
    BatchSpec("B-1004", 400, 0.25, severe_burst=(150, 154)),  # machine fault -> severe cluster
    BatchSpec("B-1005", 400, 1.60),
)

DEFAULT_SUPPLIERS: Tuple[Supplier, ...] = (
    Supplier("SUP-A", "Alt Supplier A (Faisalabad)", "cotton_lint", 2.05, 6, 0.93, "https://api.supplier-a.mock.invalid"),
    Supplier("SUP-B", "Alt Supplier B (Multan)", "cotton_lint", 1.92, 11, 0.88, "https://api.supplier-b.mock.invalid"),
    Supplier("SUP-C", "Alt Supplier C (Karachi port stock)", "cotton_lint", 2.30, 3, 0.97, "https://api.supplier-c.mock.invalid"),
)


@dataclass(frozen=True)
class PlatformConfig:
    output_dir: Path
    seed: int = 42
    workers: int = 3
    log_level: str = "INFO"
    data_source: str = "synthetic"
    prices_csv: Optional[Path] = None
    polyester_csv: Optional[Path] = None
    cotton_notional_usd: float = 2_000_000.0
    polyester_notional_usd: float = 1_500_000.0
    horizon_days: int = 1
    tariff_rate: float = 0.20
    tariff_direction: int = -1
    hedge_budget_usd: float = 2_000_000.0
    frequency: str = "auto"
    min_observations: Optional[int] = None
    detector: str = "simulator"  # "simulator" or "yolo"
    weights: Optional[Path] = None
    images_dir: Optional[Path] = None
    frame_width_m: Optional[float] = None  # physical width of fabric visible in one image
    frame_length_m: Optional[float] = None  # physical length of fabric visible in one image
    class_map: str = "figshare"
    box_limits: Optional[Path] = None  # box_limits.json from `textile-risk-train box-limits`
    detect_conf: float = 0.25
    imgsz: int = 640


class TextileRiskPlatform:
    def __init__(self, config: PlatformConfig) -> None:
        self._cfg = config
        self._out = config.output_dir
        self._geo = FrameGeometry()
        self._qcfg = QualityConfig()

    # ---- Modules 1 + 2 -------------------------------------------------------
    def _build_event_pipeline(self) -> "Tuple[InMemoryEventBroker, SupplierHedgeOrchestrator]":
        broker = InMemoryEventBroker(num_workers=self._cfg.workers)
        detector = ConsecutiveSevereDefectDetector(broker, threshold=3)
        hedger = SupplierHedgeOrchestrator(broker, DEFAULT_SUPPLIERS, MockSupplierApiClient(),
                                           HedgeConfig(), self._out / "hedge_requests")
        broker.subscribe(EventType.DEFECT_DETECTED, detector.handle)
        broker.subscribe(EventType.CRITICAL_DEFECT, hedger.handle)
        broker.subscribe(EventType.HEDGE_EXECUTED,
                         lambda e: LOGGER.info("Hedge ack for batch %s: %s", e.payload["batch_id"], e.payload["hedge_id"]))
        return broker, hedger

    def run_image_folder(self, batch_id: str = "B-IMG") -> Dict[str, Any]:
        """Run a trained YOLOv8 model over a folder of images (one image = one frame of fabric)."""
        from .detector import YoloV8Detector
        cfg = self._cfg
        if cfg.weights is None or cfg.images_dir is None or cfg.frame_width_m is None or cfg.frame_length_m is None:
            raise ValueError("--weights, --images-dir, --frame-width-m and --frame-length-m are required for --detector yolo")
        paths = sorted(p for p in cfg.images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT)
        if not paths:
            raise ValueError(f"No images found in {cfg.images_dir}")
        model = YoloV8Detector(cfg.weights, conf=cfg.detect_conf, imgsz=cfg.imgsz)
        class_map = parse_class_map(cfg.class_map)
        limits = load_box_limits(cfg.box_limits) if cfg.box_limits else None
        missing = sorted(set(model.class_names) - set(class_map))
        if missing:
            raise ValueError(f"Model classes {missing} have no entry in --class-map {cfg.class_map!r}")
        scorer = BatchQualityScorer(self._qcfg)
        broker, hedger = self._build_event_pipeline()
        broker.start()
        ingestors: Dict[Tuple[int, int], DetectionIngestor] = {}
        all_dets: List[Detection] = []
        try:
            for frame_id, path in enumerate(paths):
                raw, (w, h) = model.detect(path)
                ing = ingestors.setdefault((w, h), DetectionIngestor(
                    FrameGeometry(w, h, cfg.frame_width_m, cfg.frame_length_m), self._qcfg, class_map, limits))
                for seq_det in ing.ingest(batch_id, frame_id, raw):
                    broker.publish(Event(EventType.DEFECT_DETECTED, {
                        "batch_id": batch_id, "seq": len(all_dets), "frame_id": frame_id,
                        "defect_type": seq_det.defect_type.name, "confidence": round(seq_det.confidence, 4),
                        "severity": round(seq_det.severity, 6), "is_severe": seq_det.is_severe}))
                    all_dets.append(seq_det)
            broker.drain()
            report = scorer.score(batch_id, all_dets, len(paths) * cfg.frame_length_m)
            atomic_write_json(self._out / "batch_reports" / f"{batch_id}.json", report)
        finally:
            broker.stop()
        rejections: Dict[str, int] = {}
        for ing in ingestors.values():
            for k, v in ing.rejections.items():
                rejections[k] = rejections.get(k, 0) + v
        LOGGER.info("Image batch %s: %d images, %d detections, rejections=%s", batch_id, len(paths), len(all_dets), rejections)
        return {"batch_reports": [report], "broker_stats": broker.stats, "dead_letters": len(broker.dead_letters),
                "ingestion_rejections": rejections, "hedge_files": [str(p) for p in hedger.executed_files],
                "images_processed": len(paths)}

    def run_quality_and_events(self, batches: Sequence[BatchSpec] = DEFAULT_BATCHES) -> Dict[str, Any]:
        sim = YoloV8Simulator(self._geo, self._cfg.seed)
        ingestor = DetectionIngestor(self._geo, self._qcfg)
        scorer = BatchQualityScorer(self._qcfg)
        broker, hedger = self._build_event_pipeline()
        broker.start()

        reports: List[BatchReport] = []
        try:
            for spec in batches:
                all_dets: List[Detection] = []
                seq = 0
                for frame_id in range(spec.n_frames):
                    forced = spec.severe_burst is not None and spec.severe_burst[0] <= frame_id < spec.severe_burst[1]
                    raw = sim.generate_frame(spec.defect_rate, force_severe=forced)
                    for det in ingestor.ingest(spec.batch_id, frame_id, raw):
                        all_dets.append(det)
                        broker.publish(Event(EventType.DEFECT_DETECTED, {
                            "batch_id": det.batch_id, "seq": seq, "frame_id": det.frame_id,
                            "defect_type": det.defect_type.name, "confidence": round(det.confidence, 4),
                            "severity": round(det.severity, 6), "is_severe": det.is_severe}))
                        seq += 1
                broker.drain()
                report = scorer.score(spec.batch_id, all_dets, spec.n_frames * self._geo.frame_length_m)
                atomic_write_json(self._out / "batch_reports" / f"{spec.batch_id}.json", report)
                reports.append(report)
        finally:
            broker.stop()
        LOGGER.info("Ingestion rejections: %s", dict(ingestor.rejections))
        return {"batch_reports": reports, "broker_stats": broker.stats,
                "dead_letters": len(broker.dead_letters),
                "ingestion_rejections": dict(ingestor.rejections),
                "hedge_files": [str(p) for p in hedger.executed_files]}

    # ---- Module 3 ------------------------------------------------------------
    def _load_market(self) -> MarketData:
        cfg = self._cfg
        if cfg.data_source == "csv":
            if cfg.prices_csv is None:
                raise ValueError("--prices-csv is required for --source csv")
            return MarketDataProvider.from_csv(cfg.prices_csv, cfg.frequency)
        if cfg.data_source == "yahoo":
            if cfg.polyester_csv is None:
                raise ValueError("--polyester-csv is required for --source yahoo")
            return MarketDataProvider.from_yahoo_plus_polyester_csv(cfg.polyester_csv)
        LOGGER.warning("Using SYNTHETIC market data - results are for demonstration only")
        return MarketDataProvider.synthetic(seed=cfg.seed)

    def run_risk_engine(self) -> Path:
        cfg = self._cfg
        market = self._load_market()
        LOGGER.info("Market data: %s | %d rows | %s -> %s", market.source, len(market.prices),
                    market.prices.index[0].date(), market.prices.index[-1].date())
        engine = HistoricalVaREngine(market.log_returns,
                                     {COTTON_COL: cfg.cotton_notional_usd, POLYESTER_COL: cfg.polyester_notional_usd},
                                     min_observations=cfg.min_observations or MIN_OBSERVATIONS[market.frequency])
        if not engine.cross_check_with_pandas(cfg.horizon_days):
            raise RuntimeError("VaR cross-check failed")
        base_returns = engine.horizon_returns(cfg.horizon_days)
        baseline = engine.metrics(engine.pnl(base_returns))
        LOGGER.info("Baseline %dd historical: VaR95 $%.2f | VaR99 $%.2f | ES95 $%.2f | ES99 $%.2f",
                    cfg.horizon_days, baseline.var_95, baseline.var_99, baseline.es_95, baseline.es_99)

        spec = TariffShockSpec(tariff_rate=cfg.tariff_rate, direction=cfg.tariff_direction)
        stress = TariffStressTester(engine, spec).run(cfg.horizon_days)
        constraints = HedgeConstraints(hedge_budget_usd=cfg.hedge_budget_usd)
        solution = HedgeOptimizer(engine, constraints).optimise(stress.stressed_log_returns)

        writer = HedgeConfigWriter()
        config = writer.build(market=market, horizon_days=cfg.horizon_days, engine=engine, baseline=baseline,
                              stress=stress, spec=spec, solution=solution, constraints=constraints,
                              base_returns=base_returns)
        path = self._out / "hedge_config.json"
        writer.write(config, path)
        return path

    def run(self) -> None:
        configure_logging(self._out / "logs", self._cfg.log_level)
        t0 = time.perf_counter()
        quality = self.run_image_folder() if self._cfg.detector == "yolo" else self.run_quality_and_events()
        config_path = self.run_risk_engine()
        summary = {"completed_at": utc_now_iso(), "elapsed_s": round(time.perf_counter() - t0, 3),
                   "hedge_config": str(config_path), **quality}
        atomic_write_json(self._out / "run_summary.json", summary)
        LOGGER.info("Run complete in %.2fs. Summary: %s", summary["elapsed_s"], self._out / "run_summary.json")

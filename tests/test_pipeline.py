import json

from textile_risk.cli import parse_args
from textile_risk.orchestrator import BatchSpec, PlatformConfig, TextileRiskPlatform


def test_end_to_end_run_creates_all_artefacts(tmp_path):
    TextileRiskPlatform(PlatformConfig(output_dir=tmp_path, seed=42)).run()
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert len(summary["batch_reports"]) == 5 and summary["dead_letters"] == 0
    assert (tmp_path / "hedge_config.json").exists() and (tmp_path / "logs" / "pipeline.log").exists()
    reports = {r["batch_id"]: r for r in summary["batch_reports"]}
    assert reports["B-1005"]["downvalued"] and reports["B-1005"]["price_multiplier"] < 1.0
    assert not reports["B-1001"]["downvalued"]
    # the forced severe burst in B-1004 must produce a hedge request file
    assert any("B-1004" in f for f in summary["hedge_files"])
    for f in summary["hedge_files"]:
        assert json.loads(open(f).read())["quotes"]


def test_run_is_reproducible_for_same_seed(tmp_path):
    out = []
    for i in range(2):
        d = tmp_path / str(i)
        platform = TextileRiskPlatform(PlatformConfig(output_dir=d, seed=11))
        res = platform.run_quality_and_events()
        out.append([(r.batch_id, round(r.quality_score, 9), r.total_defects) for r in res["batch_reports"]])
    assert out[0] == out[1]


def test_cli_parses_arguments(tmp_path):
    cfg = parse_args(["--output-dir", str(tmp_path), "--seed", "5", "--horizon-days", "10",
                      "--tariff-direction", "1", "--hedge-budget", "1000000"])
    assert cfg.seed == 5 and cfg.horizon_days == 10 and cfg.tariff_direction == 1
    assert cfg.hedge_budget_usd == 1_000_000.0

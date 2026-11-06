import json
import math

import numpy as np
import pandas as pd
import pytest

from textile_risk.risk import (COTTON_COL, POLYESTER_COL, HedgeConfigWriter, HedgeConstraints,
                               HedgeOptimizer, HistoricalVaREngine, MarketDataProvider,
                               TariffShockSpec, TariffStressTester)


def test_var_and_es_hand_computed():
    pnl = np.arange(-100.0, 101.0)  # 201 points, -100..100
    assert HistoricalVaREngine.var(pnl, 0.95) == pytest.approx(90.0)
    assert HistoricalVaREngine.var(pnl, 0.99) == pytest.approx(98.0)
    assert HistoricalVaREngine.expected_shortfall(pnl, 0.95) == pytest.approx(95.0)  # mean(-100..-90)
    assert HistoricalVaREngine.expected_shortfall(pnl, 0.99) == pytest.approx(99.0)  # mean(-100..-98)


def test_pnl_uses_simple_returns_of_log_returns(engine):
    r = np.array([[math.log(1.1), math.log(0.9)]])
    assert engine.pnl(r)[0] == pytest.approx(2e6 * 0.1 + 1.5e6 * -0.1)


def test_metrics_ordering_and_cross_check(engine):
    m = engine.metrics(engine.pnl(engine.horizon_returns(1)))
    assert 0 < m.var_95 < m.var_99 <= m.es_99
    assert m.es_95 >= m.var_95
    assert engine.cross_check_with_pandas(1) and engine.cross_check_with_pandas(5)


def test_horizon_returns_are_overlapping_sums(engine):
    h1, h5 = engine.horizon_returns(1), engine.horizon_returns(5)
    assert h5.shape[0] == h1.shape[0] - 4
    np.testing.assert_allclose(h5[0], h1[:5].sum(axis=0))


def test_engine_input_validation(market):
    with pytest.raises(ValueError):
        HistoricalVaREngine(market.log_returns.iloc[:100], {COTTON_COL: 1.0})
    with pytest.raises(ValueError):
        HistoricalVaREngine(market.log_returns, {"nonexistent": 1.0})


def test_tariff_shock_vector_and_instant_pnl_exact(engine):
    spec = TariffShockSpec(tariff_rate=0.20, direction=-1)
    res = TariffStressTester(engine, spec).run(1)
    s_cot, s_pol = -math.log(1 + 0.2 * 0.35), -math.log(1 + 0.2 * 0.60)
    assert res.shock_log_returns[COTTON_COL] == pytest.approx(s_cot)
    assert res.shock_log_returns[POLYESTER_COL] == pytest.approx(s_pol)
    expected = 2e6 * (math.exp(s_cot) - 1) + 1.5e6 * (math.exp(s_pol) - 1)
    assert res.instantaneous_shock_pnl == pytest.approx(expected)


def test_stress_increases_risk_and_preserves_mean_shift(engine):
    base = engine.horizon_returns(1)
    res = TariffStressTester(engine, TariffShockSpec()).run(1)
    base_m = engine.metrics(engine.pnl(base))
    assert res.stressed_metrics.var_99 > base_m.var_99
    np.testing.assert_allclose(res.stressed_log_returns.mean(axis=0),
                               base.mean(axis=0) + np.array(list(res.shock_log_returns.values())), atol=1e-12)


def test_stress_rejects_bad_direction(engine):
    with pytest.raises(ValueError):
        TariffStressTester(engine, TariffShockSpec(direction=0))


def test_lp_matches_brute_force_grid(engine):
    stress = TariffStressTester(engine, TariffShockSpec()).run(1)
    cons = HedgeConstraints()
    sol = HedgeOptimizer(engine, cons).optimise(stress.stressed_log_returns)
    S = np.exp(stress.stressed_log_returns) - 1

    def obj(y1, y2):
        p = S @ np.array([2e6 - y1, 1.5e6 - y2])
        q = np.quantile(p, 0.01)
        return -p[p <= q].mean() + cons.transaction_cost_bps / 1e4 * (y1 + y2)

    grid = min(obj(a, b) for a in np.linspace(0, 1.8e6, 181) for b in np.linspace(0, 1.35e6, 136)
               if a + b <= cons.hedge_budget_usd + 1)
    # LP is the exact optimum so it can never be worse than the grid; the small gap is the
    # fractional-tail (Rockafellar-Uryasev) vs discrete-tail ES estimator difference.
    assert sol.objective_es_usd <= grid * (1 + 1e-3)
    assert sol.objective_es_usd == pytest.approx(grid, rel=1e-3)
    assert sol.stressed_after.es_99 < sol.stressed_before.es_99


def test_hedge_respects_constraints(engine):
    stress = TariffStressTester(engine, TariffShockSpec()).run(1)
    cons = HedgeConstraints(max_hedge_ratio=0.5, hedge_budget_usd=1_000_000.0)
    sol = HedgeOptimizer(engine, cons).optimise(stress.stressed_log_returns)
    h = sol.hedge_notionals
    assert sum(abs(v) for v in h.values()) <= 1_000_000.0 + 1e-3
    assert abs(h[COTTON_COL]) <= 0.5 * 2e6 + 1e-3 and abs(h[POLYESTER_COL]) <= 0.5 * 1.5e6 + 1e-3
    assert all(v <= 1e-9 for v in h.values())  # long exposures are only ever hedged short


def _build_config(market, engine):
    spec, cons = TariffShockSpec(), HedgeConstraints()
    base = engine.horizon_returns(1)
    stress = TariffStressTester(engine, spec).run(1)
    sol = HedgeOptimizer(engine, cons).optimise(stress.stressed_log_returns)
    return HedgeConfigWriter().build(market=market, horizon_days=1, engine=engine,
                                     baseline=engine.metrics(engine.pnl(base)), stress=stress, spec=spec,
                                     solution=sol, constraints=cons, base_returns=base)


def test_config_roundtrip_and_tamper_detection(market, engine, tmp_path):
    cfg = _build_config(market, engine)
    path = tmp_path / "hedge_config.json"
    HedgeConfigWriter().write(cfg, path)
    data = json.loads(path.read_text())
    assert data["data"]["data_quality"] == "SYNTHETIC_DEMO_DATA_NOT_FOR_TRADING"
    cot = next(a for a in data["hedge_adjustments"] if a["instrument"] == COTTON_COL)
    assert cot["contracts_rounded"] == round(abs(cot["hedge_notional_usd"]) / cot["contract_notional_usd"])
    assert next(a for a in data["hedge_adjustments"] if a["instrument"] == POLYESTER_COL)["contracts_rounded"] is None
    data["hedge_adjustments"][0]["hedge_notional_usd"] = 1.0  # tamper
    path.write_text(json.dumps(data))
    from textile_risk.common import canonical_sha256
    body = {k: v for k, v in json.loads(path.read_text()).items() if k != "integrity"}
    assert canonical_sha256(body) != data["integrity"]["sha256"]


def test_validator_blocks_sign_flip(market, engine):
    cfg = _build_config(market, engine)
    cfg.pop("integrity")
    cfg["hedge_adjustments"][0]["post_hedge_exposure_usd"] = -1.0
    with pytest.raises(ValueError):
        HedgeConfigWriter.validate(cfg)


def test_csv_loader_ok_and_rejects_bad_data(tmp_path):
    good = tmp_path / "p.csv"
    pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=5),
                  "Cotton_Futures": [70, 71, 70.5, 72, 71.5],
                  "Polyester_Spot": [0.9, 0.91, 0.9, 0.92, 0.93]}).to_csv(good, index=False)
    assert len(MarketDataProvider.from_csv(good).prices) == 5
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-01"] * 1 + ["2024-01-02", "2024-01-03"],
                  "cotton_futures": [70, -1, 71], "polyester_spot": [0.9, 0.9, 0.9]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        MarketDataProvider.from_csv(bad)
    missing = tmp_path / "m.csv"
    pd.DataFrame({"date": ["2024-01-01"], "cotton_futures": [70]}).to_csv(missing, index=False)
    with pytest.raises(ValueError):
        MarketDataProvider.from_csv(missing)


def test_synthetic_is_reproducible_and_positive():
    a, b = MarketDataProvider.synthetic(seed=3), MarketDataProvider.synthetic(seed=3)
    pd.testing.assert_frame_equal(a.prices, b.prices)
    assert (a.prices > 0).all().all() and a.is_synthetic and len(a.prices) == 1500

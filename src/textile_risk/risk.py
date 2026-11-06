"""Module 3 - Historical VaR/ES, tariff stress test, LP hedge optimiser, config writer."""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from .common import atomic_write_json, canonical_sha256, utc_now_iso

LOGGER = logging.getLogger("textile.risk")


COTTON_COL = "cotton_futures"  # ICE Cotton No.2, quoted in US cents per lb
POLYESTER_COL = "polyester_spot"  # USD per kg


@dataclass(frozen=True)
class InstrumentSpec:
    column: str
    label: str
    contract_size: Optional[float]  # units per contract, None if OTC / spot
    price_to_usd_per_unit: float  # converts quoted price to USD per unit

    def contract_notional_usd(self, price: float) -> Optional[float]:
        if self.contract_size is None:
            return None
        return price * self.price_to_usd_per_unit * self.contract_size


INSTRUMENTS: Dict[str, InstrumentSpec] = {
    COTTON_COL: InstrumentSpec(COTTON_COL, "ICE Cotton No.2 Futures", 50_000.0, 0.01),
    POLYESTER_COL: InstrumentSpec(POLYESTER_COL, "Polyester Spot (OTC forward overlay)", None, 1.0),
}


COLUMN_ALIASES: Dict[str, tuple] = {
    COTTON_COL: (COTTON_COL, "cotton_global_usc_per_lb"),
    POLYESTER_COL: (POLYESTER_COL, "polyester_ppi_index"),
}
SERIES_CAVEATS: Dict[str, str] = {
    "cotton_global_usc_per_lb": "IMF/Cotlook 'A' monthly benchmark (US cents/lb), not ICE futures settlements",
    "polyester_ppi_index": "BLS producer price index (Jun 2011 = 100), a smoothed US producer price, not a traded spot price",
}
MIN_OBSERVATIONS: Dict[str, int] = {"daily": 250, "weekly": 104, "monthly": 120}
PERIOD_UNIT: Dict[str, str] = {"daily": "day", "weekly": "week", "monthly": "month"}
VALID_FREQUENCIES = ("auto", "daily", "weekly", "monthly")


def detect_frequency(index: pd.DatetimeIndex) -> str:
    """Classify an index by its median spacing: <=4 days daily, <=10 weekly, else monthly."""
    if len(index) < 3:
        raise ValueError("Need at least 3 dates to detect frequency")
    median_gap = float(np.median(np.diff(index.values).astype("timedelta64[D]").astype(float)))
    if median_gap <= 4.0:
        return "daily"
    return "weekly" if median_gap <= 10.0 else "monthly"


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    source: str
    is_synthetic: bool
    frequency: str = "daily"
    source_columns: Mapping[str, str] = field(default_factory=dict)

    @property
    def log_returns(self) -> pd.DataFrame:
        return np.log(self.prices / self.prices.shift(1)).dropna()


class MarketDataProvider:
    """Loads real price history (CSV / Yahoo) or a clearly flagged synthetic series."""

    @staticmethod
    def _validate(prices: pd.DataFrame, source: str) -> pd.DataFrame:
        missing = [c for c in (COTTON_COL, POLYESTER_COL) if c not in prices.columns]
        if missing:
            raise ValueError(f"{source}: missing required columns {missing}")
        prices = prices[[COTTON_COL, POLYESTER_COL]].astype("float64").sort_index()
        prices = prices[~prices.index.duplicated(keep="last")]
        n_nan = int(prices.isna().sum().sum())
        if n_nan:
            LOGGER.warning("%s: %d missing prices, forward-filling (limit 3) then dropping", source, n_nan)
            prices = prices.ffill(limit=3).dropna()
        if (prices <= 0).any().any() or not np.isfinite(prices.to_numpy()).all():
            raise ValueError(f"{source}: prices must be finite and strictly positive")
        if len(prices) < 3:
            raise ValueError(f"{source}: need at least 3 price rows")
        return prices

    @staticmethod
    def _resolve_columns(df: pd.DataFrame, source: str) -> "tuple[pd.DataFrame, Dict[str, str]]":
        """Map accepted column aliases onto canonical names; remember what the file really called them."""
        used: Dict[str, str] = {}
        out = pd.DataFrame(index=df.index)
        for canonical, aliases in COLUMN_ALIASES.items():
            found = next((a for a in aliases if a in df.columns), None)
            if found is None:
                raise ValueError(f"{source}: need one of {list(aliases)} for {canonical}; "
                                 f"file has columns {list(df.columns)}")
            out[canonical] = df[found]
            used[canonical] = found
        return out, used

    @classmethod
    def from_csv(cls, path: Path, frequency: str = "auto") -> MarketData:
        if frequency not in VALID_FREQUENCIES:
            raise ValueError(f"frequency must be one of {VALID_FREQUENCIES}")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" not in df.columns:
            raise ValueError(f"{path}: a 'date' column is required")
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df, used = cls._resolve_columns(df.set_index("date"), str(path))
        prices = cls._validate(df, str(path))
        freq = detect_frequency(prices.index) if frequency == "auto" else frequency
        LOGGER.info("Loaded %s: %d rows, frequency=%s, columns=%s", path, len(prices), freq, used)
        return MarketData(prices, f"csv:{path}", False, freq, used)

    @classmethod
    def from_yahoo_plus_polyester_csv(cls, polyester_csv: Path, start: str = "2019-01-01") -> MarketData:
        try:
            import yfinance as yf  # optional dependency
        except ImportError as exc:
            raise RuntimeError("pip install yfinance to use --source yahoo") from exc
        raw = yf.download("CT=F", start=start, auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError("Yahoo Finance returned no data for CT=F")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        cotton = close.rename(COTTON_COL)
        cotton.index = pd.to_datetime(cotton.index).tz_localize(None).normalize()
        poly_df = pd.read_csv(polyester_csv)
        poly_df.columns = [c.strip().lower() for c in poly_df.columns]
        poly_df["date"] = pd.to_datetime(poly_df["date"])
        poly = poly_df.set_index("date")[POLYESTER_COL]
        joined = pd.concat([cotton, poly], axis=1, join="inner")
        return MarketData(cls._validate(joined, "yahoo+csv"), "yahoo:CT=F + polyester csv", False, "daily",
                          {COTTON_COL: "yahoo:CT=F", POLYESTER_COL: str(polyester_csv)})

    @classmethod
    def synthetic(cls, n_days: int = 1500, seed: int = 7) -> MarketData:
        """Correlated Student-t(5) returns, annual vol 24% / 16%, correlation 0.45.
        FOR OFFLINE DEMOS ONLY - never use for real risk numbers."""
        rng = np.random.default_rng(seed)
        df_t = 5.0
        corr = np.array([[1.0, 0.45], [0.45, 1.0]])
        chol = np.linalg.cholesky(corr)
        z = rng.standard_t(df_t, size=(n_days, 2)) / math.sqrt(df_t / (df_t - 2.0))
        daily_vol = np.array([0.24, 0.16]) / math.sqrt(252.0)
        rets = (z @ chol.T) * daily_vol
        start = np.array([72.0, 0.95])
        levels = start * np.exp(np.cumsum(rets, axis=0))
        idx = pd.bdate_range(start="2020-01-01", periods=n_days)
        prices = pd.DataFrame(levels, index=idx, columns=[COTTON_COL, POLYESTER_COL])
        return MarketData(cls._validate(prices, "synthetic"), "synthetic", True)


@dataclass(frozen=True)
class RiskMetrics:
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    mean_pnl: float
    worst_pnl: float
    n_scenarios: int


class HistoricalVaREngine:
    """Historical-simulation VaR / ES on a signed-notional book.
    P&L_s = sum_i N_i * (exp(r_{s,i}) - 1), r = h-day overlapping log returns.
    VaR_c = -Quantile_{1-c}(P&L); ES_c = -E[P&L | P&L <= Quantile_{1-c}(P&L)]."""

    def __init__(self, log_returns: pd.DataFrame, notionals: Mapping[str, float],
                 min_observations: int = 250) -> None:
        self.assets: List[str] = list(notionals.keys())
        missing = [a for a in self.assets if a not in log_returns.columns]
        if missing:
            raise ValueError(f"No return series for {missing}")
        if len(log_returns) < min_observations:
            raise ValueError(f"Need >= {min_observations} observations, got {len(log_returns)}")
        self._returns = log_returns[self.assets]
        self.notionals = np.array([notionals[a] for a in self.assets], dtype=np.float64)

    def horizon_returns(self, horizon_days: int) -> np.ndarray:
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        window = self._returns.rolling(horizon_days).sum().dropna() if horizon_days > 1 else self._returns
        return window.to_numpy(dtype=np.float64)

    def pnl(self, log_ret: np.ndarray, notionals: Optional[np.ndarray] = None) -> np.ndarray:
        n = self.notionals if notionals is None else notionals
        return (np.exp(log_ret) - 1.0) @ n

    @staticmethod
    def var(pnl: np.ndarray, confidence: float) -> float:
        return float(-np.quantile(pnl, 1.0 - confidence, method="linear"))

    @staticmethod
    def expected_shortfall(pnl: np.ndarray, confidence: float) -> float:
        q = np.quantile(pnl, 1.0 - confidence, method="linear")
        return float(-pnl[pnl <= q].mean())

    def metrics(self, pnl: np.ndarray) -> RiskMetrics:
        return RiskMetrics(var_95=self.var(pnl, 0.95), var_99=self.var(pnl, 0.99),
                           es_95=self.expected_shortfall(pnl, 0.95),
                           es_99=self.expected_shortfall(pnl, 0.99),
                           mean_pnl=float(pnl.mean()), worst_pnl=float(pnl.min()),
                           n_scenarios=int(pnl.size))

    def cross_check_with_pandas(self, horizon_days: int, tol: float = 1e-6) -> bool:
        """Independent recomputation through pandas to guard against indexing mistakes."""
        win = self._returns.rolling(horizon_days).sum().dropna() if horizon_days > 1 else self._returns
        pnl_s = ((np.exp(win) - 1.0) * pd.Series(self.notionals, index=self.assets)).sum(axis=1)
        ok = all(abs(-pnl_s.quantile(1 - c) - self.var(self.pnl(win.to_numpy()), c)) <= tol
                 for c in (0.95, 0.99))
        LOGGER.info("VaR pandas cross-check: %s", "PASS" if ok else "FAIL")
        return ok


@dataclass(frozen=True)
class TariffShockSpec:
    """tariff_rate=0.20 -> 20% import tariff. Price response of asset i (log-return):
        s_i = direction * ln(1 + tariff_rate * pass_through_i)
    direction=-1: world benchmark falls (import demand destruction); +1: domestic price spike.
    Stressed scenarios:  R' = mu + vol_multiplier * (R - mu) + s."""

    tariff_rate: float = 0.20
    pass_through: Mapping[str, float] = field(
        default_factory=lambda: {COTTON_COL: 0.35, POLYESTER_COL: 0.60})
    direction: int = -1
    vol_multiplier: float = 1.5


@dataclass(frozen=True)
class StressResult:
    shock_log_returns: Dict[str, float]
    instantaneous_shock_pnl: float
    stressed_metrics: RiskMetrics
    stressed_log_returns: np.ndarray


class TariffStressTester:
    def __init__(self, engine: HistoricalVaREngine, spec: TariffShockSpec) -> None:
        if spec.direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        self._engine = engine
        self._spec = spec

    def run(self, horizon_days: int) -> StressResult:
        base = self._engine.horizon_returns(horizon_days)
        mu = base.mean(axis=0)
        shock = np.array([self._spec.direction * math.log1p(
            self._spec.tariff_rate * self._spec.pass_through[a]) for a in self._engine.assets])
        stressed = mu + self._spec.vol_multiplier * (base - mu) + shock
        pnl = self._engine.pnl(stressed)
        instant = float(((np.exp(shock) - 1.0) * self._engine.notionals).sum())
        result = StressResult(dict(zip(self._engine.assets, shock.tolist())), instant,
                              self._engine.metrics(pnl), stressed)
        LOGGER.info("Tariff stress (%.0f%%): instantaneous P&L $%.2f | stressed VaR95 $%.2f VaR99 $%.2f",
                    self._spec.tariff_rate * 100, instant,
                    result.stressed_metrics.var_95, result.stressed_metrics.var_99)
        return result


@dataclass(frozen=True)
class HedgeConstraints:
    max_hedge_ratio: float = 0.90  # |hedge_i| <= ratio * |exposure_i|
    hedge_budget_usd: float = 2_000_000.0  # cap on sum |hedge_i| (margin / credit line)
    transaction_cost_bps: float = 8.0
    es_confidence: float = 0.99


@dataclass(frozen=True)
class HedgeSolution:
    hedge_notionals: Dict[str, float]  # signed: <0 short, >0 long
    objective_es_usd: float
    stressed_before: RiskMetrics
    stressed_after: RiskMetrics
    solver_status: str


class HedgeOptimizer:
    """
    Minimises  ES_c(stressed net P&L) + cost  over hedge overlays h_i (Rockafellar-Uryasev LP).
    A hedge only reduces an existing exposure (sign(h_i) = -sign(N_i), |h_i| <= ratio*|N_i|), so
    the optimiser cannot turn into a directional bet on the tariff scenario.

        min   alpha + 1/((1-c)K) * sum_k u_k + (bps/1e4) * sum_i y_i
        s.t.  u_k >= -pnl_k(y) - alpha,  u_k >= 0,  0 <= y_i <= ratio*|N_i|,  sum_i y_i <= budget
        with  h_i = -sign(N_i) * y_i,  pnl_k = sum_i (N_i + h_i) * S_{k,i},  S = exp(R') - 1
    """

    def __init__(self, engine: HistoricalVaREngine, constraints: HedgeConstraints) -> None:
        self._engine = engine
        self._c = constraints

    def optimise(self, stressed_log_returns: np.ndarray) -> HedgeSolution:
        eng, cons = self._engine, self._c
        n = len(eng.assets)
        simple = np.exp(stressed_log_returns) - 1.0
        k = simple.shape[0]
        sign = np.sign(eng.notionals)
        base_pnl_k = simple @ eng.notionals

        a_y = simple * sign  # coefficient of y_i in the constraint row
        a_ub = sparse.hstack([sparse.csr_matrix(a_y), -sparse.csr_matrix(np.ones((k, 1))),
                              -sparse.identity(k, format="csr")], format="csr")
        b_ub = base_pnl_k
        budget_row = sparse.csr_matrix(np.concatenate([np.ones(n), np.zeros(1 + k)]).reshape(1, -1))
        a_ub = sparse.vstack([a_ub, budget_row], format="csr")
        b_ub = np.concatenate([b_ub, [cons.hedge_budget_usd]])

        cost = cons.transaction_cost_bps / 1e4
        c_vec = np.concatenate([np.full(n, cost), [1.0], np.full(k, 1.0 / ((1.0 - cons.es_confidence) * k))])
        bounds = ([(0.0, cons.max_hedge_ratio * abs(N)) for N in eng.notionals]
                  + [(None, None)] + [(0.0, None)] * k)
        res = linprog(c_vec, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"Hedge LP failed: {res.message}")

        y = res.x[:n]
        hedge = -sign * y
        before = eng.metrics(base_pnl_k)
        after = eng.metrics(simple @ (eng.notionals + hedge))
        LOGGER.info("Hedge LP solved (%s): objective $%.2f | stressed ES99 $%.2f -> $%.2f",
                    res.message, res.fun, before.es_99, after.es_99)
        return HedgeSolution(dict(zip(eng.assets, hedge.tolist())), float(res.fun), before, after, res.message)


class HedgeConfigWriter:
    """Builds, validates and persists the production hedge-adjustment JSON config."""

    SCHEMA_VERSION = "1.0.0"

    @staticmethod
    def _r(x: float, nd: int = 6) -> float:
        return round(float(x), nd)

    def build(self, *, market: MarketData, horizon_days: int, engine: HistoricalVaREngine,
              baseline: RiskMetrics, stress: StressResult, spec: TariffShockSpec,
              solution: HedgeSolution, constraints: HedgeConstraints, base_returns: np.ndarray
              ) -> Dict[str, Any]:
        gross = float(np.abs(engine.notionals).sum())
        last_prices = market.prices.iloc[-1]
        post_pnl = engine.pnl(base_returns, engine.notionals + np.array(
            [solution.hedge_notionals[a] for a in engine.assets]))
        post_base = engine.metrics(post_pnl)
        adjustments: List[Dict[str, Any]] = []
        for asset, exposure in zip(engine.assets, engine.notionals):
            h = solution.hedge_notionals[asset]
            spec_i = INSTRUMENTS[asset]
            per_contract = spec_i.contract_notional_usd(float(last_prices[asset]))
            adjustments.append({
                "instrument": asset, "description": spec_i.label,
                "action": "SHORT" if h < -1e-6 else ("LONG" if h > 1e-6 else "NONE"),
                "hedge_notional_usd": self._r(h, 2),
                "hedge_weight_of_gross": self._r(h / gross),
                "pre_hedge_exposure_usd": self._r(exposure, 2),
                "post_hedge_exposure_usd": self._r(exposure + h, 2),
                "pre_hedge_weight": self._r(exposure / gross),
                "post_hedge_weight": self._r((exposure + h) / gross),
                "hedge_ratio": self._r(abs(h) / abs(exposure)) if exposure else 0.0,
                "last_price": self._r(float(last_prices[asset]), 6),
                "contract_notional_usd": None if per_contract is None else self._r(per_contract, 2),
                "contracts_rounded": None if per_contract is None else int(round(abs(h) / per_contract)),
            })

        def m(x: RiskMetrics) -> Dict[str, float]:
            return {k: self._r(v, 2) if k != "n_scenarios" else int(v) for k, v in asdict(x).items()}

        n_sc = baseline.n_scenarios
        unit = PERIOD_UNIT[market.frequency]
        warnings: List[str] = []
        if market.is_synthetic:
            warnings.append("Synthetic data: figures are illustrative only")
        if market.frequency != "daily":
            warnings.append(f"Returns are {market.frequency}: VaR/ES horizon is {horizon_days} {unit}(s), not days")
        if n_sc * 0.01 < 10:
            warnings.append(f"99% VaR/ES rests on only {n_sc * 0.01:.1f} tail observations: statistically unreliable")
        for col in market.source_columns.values():
            if col in SERIES_CAVEATS:
                warnings.append(f"{col}: {SERIES_CAVEATS[col]}")
        body: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "run_id": uuid.uuid4().hex,
            "data": {"source": market.source,
                     "data_quality": "SYNTHETIC_DEMO_DATA_NOT_FOR_TRADING" if market.is_synthetic else "HISTORICAL",
                     "start": str(market.prices.index[0].date()), "end": str(market.prices.index[-1].date()),
                     "n_price_rows": int(len(market.prices)), "frequency": market.frequency,
                     "source_columns": dict(market.source_columns)},
            "reliability": {"scenarios": int(n_sc), "tail_observations_95": self._r(n_sc * 0.05, 2),
                            "tail_observations_99": self._r(n_sc * 0.01, 2), "warnings": warnings},
            "risk_metrics": {"horizon_periods": horizon_days, "period_unit": unit,
                             "gross_notional_usd": self._r(gross, 2),
                             "baseline_historical": m(baseline),
                             "tariff_stress": {**m(stress.stressed_metrics),
                                               "instantaneous_shock_pnl_usd": self._r(stress.instantaneous_shock_pnl, 2)},
                             "post_hedge_stressed": m(solution.stressed_after),
                             "post_hedge_baseline_historical": m(post_base),
                             "stressed_es99_reduction_pct": self._r(
                                 100.0 * (1.0 - solution.stressed_after.es_99 / solution.stressed_before.es_99), 4)},
            "tariff_scenario": {"tariff_rate": spec.tariff_rate, "direction": spec.direction,
                                "vol_multiplier": spec.vol_multiplier,
                                "pass_through": dict(spec.pass_through),
                                "shock_log_returns": {a: self._r(v, 8) for a, v in stress.shock_log_returns.items()}},
            "optimiser": {"objective": f"min ES{int(constraints.es_confidence * 100)} (stressed) + transaction cost",
                          "method": "Rockafellar-Uryasev LP / scipy HiGHS", "status": solution.solver_status,
                          "constraints": asdict(constraints)},
            "hedge_adjustments": adjustments,
        }
        self.validate(body)
        return {**body, "integrity": {"algorithm": "sha256-canonical-json", "sha256": canonical_sha256(body)}}

    @staticmethod
    def validate(body: Mapping[str, Any]) -> None:
        for adj in body["hedge_adjustments"]:
            pre, post, h = adj["pre_hedge_exposure_usd"], adj["post_hedge_exposure_usd"], adj["hedge_notional_usd"]
            if pre * post < -1e-6:
                raise ValueError(f"{adj['instrument']}: hedge flips exposure sign")
            if abs(post) > abs(pre) + 1e-6:
                raise ValueError(f"{adj['instrument']}: hedge increases exposure")
            if adj["action"] == "SHORT" and h > 0 or adj["action"] == "LONG" and h < 0:
                raise ValueError(f"{adj['instrument']}: action/sign mismatch")
        if sum(abs(a["hedge_notional_usd"]) for a in body["hedge_adjustments"]) \
                > body["optimiser"]["constraints"]["hedge_budget_usd"] + 1.0:
            raise ValueError("hedge budget exceeded")

    def write(self, config: Dict[str, Any], path: Path) -> None:
        atomic_write_json(path, config)
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        body = {k: v for k, v in reloaded.items() if k != "integrity"}
        if canonical_sha256(body) != reloaded["integrity"]["sha256"]:
            raise RuntimeError("Integrity check failed after write")
        LOGGER.info("Hedge config written and verified: %s", path)


import pytest

from textile_risk.quality import FrameGeometry, QualityConfig
from textile_risk.risk import COTTON_COL, POLYESTER_COL, HistoricalVaREngine, MarketDataProvider


@pytest.fixture
def geometry() -> FrameGeometry:
    return FrameGeometry()


@pytest.fixture
def qcfg() -> QualityConfig:
    return QualityConfig()


@pytest.fixture
def market():
    return MarketDataProvider.synthetic(seed=42)


@pytest.fixture
def engine(market):
    return HistoricalVaREngine(market.log_returns,
                               {COTTON_COL: 2_000_000.0, POLYESTER_COL: 1_500_000.0})

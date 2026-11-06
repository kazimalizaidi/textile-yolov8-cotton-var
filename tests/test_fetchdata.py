import urllib.error
from datetime import date

import pytest

from textile_risk.fetchdata import (COTTON_SERIES, POLYESTER_SERIES, FetchError, download_text, main, merge_series,
                                    parse_fred_csv, write_outputs)
from textile_risk.risk import MarketDataProvider


def _fred_text(series_id: str, start_year: int = 2015, months: int = 60, base: float = 70.0, header: str = "observation_date",
               missing_at: int = -1) -> str:
    lines = [f"{header},{series_id}"]
    for i in range(months):
        y, m = start_year + i // 12, i % 12 + 1
        value = "." if i == missing_at else f"{base + i * 0.37 + (i % 5) * 0.11:.5f}"
        lines.append(f"{y}-{m:02d}-01,{value}")
    return "\n".join(lines) + "\n"


def test_parse_handles_both_headers_and_missing_values():
    new = parse_fred_csv(_fred_text(COTTON_SERIES, missing_at=10), COTTON_SERIES)
    old = parse_fred_csv(_fred_text(COTTON_SERIES, header="DATE"), COTTON_SERIES)
    assert len(new) == 59 and len(old) == 60 and date(2015, 11, 1) not in new
    assert old[date(2015, 1, 1)] == pytest.approx(70.0)


@pytest.mark.parametrize("bad", [
    "date,WRONGID\n2015-01-01,1.0\n",                                   # wrong series id
    "observation_date,PCOTTINDUSDM\n2015-01-01,abc\n" + "\n".join(f"2015-{m:02d}-01,1.0" for m in range(2, 13)) + "\n",
    "observation_date,PCOTTINDUSDM\n2015-01-01,-3.0\n",                 # non-positive
    "observation_date,PCOTTINDUSDM\n2015-01-01,1.0\n2015-02-01,2.0\n",  # far too few rows
    "",
])
def test_parse_rejects_bad_files(bad):
    with pytest.raises(FetchError):
        parse_fred_csv(bad, COTTON_SERIES)


def test_merge_is_inner_join_and_sorted():
    c = parse_fred_csv(_fred_text(COTTON_SERIES, start_year=2012, months=96), COTTON_SERIES)
    p = parse_fred_csv(_fred_text(POLYESTER_SERIES, start_year=2015, months=60, base=100.0), POLYESTER_SERIES)
    merged = merge_series(c, p)
    assert len(merged) == 60 and merged[0][0] == date(2015, 1, 1) and merged == sorted(merged)


def test_outputs_are_loadable_by_the_pipeline_loader(tmp_path):
    c = parse_fred_csv(_fred_text(COTTON_SERIES, start_year=2012, months=96), COTTON_SERIES)
    p = parse_fred_csv(_fred_text(POLYESTER_SERIES, months=60, base=100.0), POLYESTER_SERIES)
    files = write_outputs(c, p, tmp_path / "data")
    assert {f.name for f in files} == {"cotton_global_imf_fred_monthly.csv", "polyester_ppi_fred_monthly.csv",
                                       "cotton_polyester_monthly_fred.csv"}
    md = MarketDataProvider.from_csv(tmp_path / "data" / "cotton_polyester_monthly_fred.csv")
    assert md.frequency == "monthly" and len(md.prices) == 60
    assert md.source_columns["cotton_futures"] == "cotton_global_usc_per_lb"
    assert md.prices["polyester_spot"].iloc[0] == pytest.approx(100.0)
    sources = (tmp_path / "data" / "SOURCES.txt").read_text()
    assert "International Monetary Fund" in sources and "DO NOT REDISTRIBUTE" in sources


def test_main_offline_files_and_failure_exit_codes(tmp_path, capsys):
    cf, pf = tmp_path / "c.csv", tmp_path / "p.csv"
    cf.write_text(_fred_text(COTTON_SERIES, months=48))
    pf.write_text(_fred_text(POLYESTER_SERIES, months=48, base=100.0))
    assert main(["--cotton-file", str(cf), "--polyester-file", str(pf), "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "cotton_polyester_monthly_fred.csv").exists()
    cf.write_text("garbage")
    assert main(["--cotton-file", str(cf), "--polyester-file", str(pf), "--out-dir", str(tmp_path / "out2")]) == 2
    assert "FAIL" in capsys.readouterr().err


def test_download_errors_give_manual_instructions():
    def offline(request, timeout):
        raise urllib.error.URLError("no network")
    with pytest.raises(FetchError, match="Download > CSV"):
        download_text(COTTON_SERIES, opener=offline)

    class Resp:
        status = 503
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b""
    with pytest.raises(FetchError, match="HTTP 503"):
        download_text(COTTON_SERIES, opener=lambda request, timeout: Resp())


def test_download_returns_text_on_200():
    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"observation_date,PCOTTINDUSDM\n"
    assert download_text(COTTON_SERIES, opener=lambda request, timeout: Resp()).startswith("observation_date")

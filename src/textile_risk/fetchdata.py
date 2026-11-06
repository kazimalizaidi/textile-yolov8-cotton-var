"""Download the monthly price series from FRED (standard library only: no pandas, no API key).

    python -m textile_risk.fetchdata                 # writes three CSVs into ./data
    python -m textile_risk.fetchdata --out-dir data

The IMF cotton series is copyrighted ("Copyrighted: Citation Required" on FRED), so this repository does not
redistribute it: every user downloads it themselves, subject to the IMF terms (https://www.imf.org/external/terms.htm).
Series: PCOTTINDUSDM (IMF, Global price of Cotton, US cents/lb) and WPU0315040147 (BLS, Producer Price Index:
Polyester Manufactured Fibers, index Jun 2011 = 100).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

COTTON_SERIES = "PCOTTINDUSDM"
POLYESTER_SERIES = "WPU0315040147"
COTTON_COL = "cotton_global_usc_per_lb"
POLYESTER_COL = "polyester_ppi_index"
URL_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

SOURCES_TEXT = """Data sources (retrieved {when})

cotton_global_usc_per_lb  - International Monetary Fund, Global price of Cotton [PCOTTINDUSDM], retrieved from FRED,
  Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PCOTTINDUSDM
  Copyright (c) International Monetary Fund. Terms of use: https://www.imf.org/external/terms.htm
  DO NOT REDISTRIBUTE this file; it is generated locally by `python -m textile_risk.fetchdata`.

polyester_ppi_index       - U.S. Bureau of Labor Statistics, Producer Price Index by Commodity: Textile Products and
  Apparel: Polyester Manufactured Fibers [WPU0315040147], retrieved from FRED, Federal Reserve Bank of St. Louis;
  https://fred.stlouisfed.org/series/WPU0315040147
"""


class FetchError(RuntimeError):
    """Raised for network problems or unexpected file contents, with a message that says what to do next."""


def parse_fred_csv(text: str, series_id: str) -> Dict[date, float]:
    """Parse FRED's graph CSV: two columns (date, series_id). Missing values ('.' or empty) are skipped."""
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) < 2 or len(rows[0]) != 2 or rows[0][1].strip() != series_id:
        raise FetchError(f"Unexpected CSV header {rows[0] if rows else None!r}; expected two columns ending in {series_id}")
    out: Dict[date, float] = {}
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            raise FetchError(f"{series_id}: malformed line {line_no}: {row!r}")
        raw = row[1].strip()
        if raw in ("", "."):
            continue
        try:
            day = datetime.strptime(row[0].strip(), "%Y-%m-%d").date()
            value = float(raw)
        except ValueError as exc:
            raise FetchError(f"{series_id}: cannot parse line {line_no}: {row!r}") from exc
        if not math.isfinite(value) or value <= 0:
            raise FetchError(f"{series_id}: non-positive or non-finite value on line {line_no}: {row!r}")
        out[day] = value
    if len(out) < 24:
        raise FetchError(f"{series_id}: only {len(out)} observations parsed; refusing to write a near-empty file")
    return out


def merge_series(cotton: Dict[date, float], polyester: Dict[date, float]) -> List[Tuple[date, float, float]]:
    """Inner join on date, sorted ascending."""
    common = sorted(set(cotton) & set(polyester))
    if len(common) < 24:
        raise FetchError(f"Only {len(common)} overlapping months between the two series")
    return [(d, cotton[d], polyester[d]) for d in common]


def _fmt(value: float) -> str:
    return repr(round(value, 6))


def write_outputs(cotton: Dict[date, float], polyester: Dict[date, float], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_series(cotton, polyester)
    files: List[Path] = []

    def write(name: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        path = out_dir / name
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)
        files.append(path)

    write("cotton_global_imf_fred_monthly.csv", ["date", COTTON_COL], [(d.isoformat(), _fmt(v)) for d, v in sorted(cotton.items()) if d >= merged[0][0]])
    write("polyester_ppi_fred_monthly.csv", ["date", POLYESTER_COL], [(d.isoformat(), _fmt(v)) for d, v in sorted(polyester.items())])
    write("cotton_polyester_monthly_fred.csv", ["date", COTTON_COL, POLYESTER_COL],
          [(d.isoformat(), _fmt(c), _fmt(p)) for d, c, p in merged])
    (out_dir / "SOURCES.txt").write_text(SOURCES_TEXT.format(when=date.today().isoformat()), encoding="utf-8")
    return files


def download_text(series_id: str, opener: Callable = urllib.request.urlopen, timeout: float = 30.0) -> str:
    url = URL_TEMPLATE.format(series_id=series_id)
    request = urllib.request.Request(url, headers={"User-Agent": "textile-risk-platform (educational use)"})
    try:
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise FetchError(f"{url} returned HTTP {status}")
            return response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise FetchError(
            f"Could not download {series_id} ({exc}). Open https://fred.stlouisfed.org/series/{series_id}, press Download > CSV, "
            f"and save it manually, then re-run with --{('cotton' if series_id == COTTON_SERIES else 'polyester')}-file <path>.") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download the FRED price series used by this project into ./data")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--cotton-file", type=Path, help="use a manually downloaded FRED CSV for PCOTTINDUSDM instead of the network")
    p.add_argument("--polyester-file", type=Path, help="use a manually downloaded FRED CSV for WPU0315040147 instead of the network")
    a = p.parse_args(argv)
    try:
        cotton_text = a.cotton_file.read_text(encoding="utf-8") if a.cotton_file else download_text(COTTON_SERIES)
        poly_text = a.polyester_file.read_text(encoding="utf-8") if a.polyester_file else download_text(POLYESTER_SERIES)
        files = write_outputs(parse_fred_csv(cotton_text, COTTON_SERIES), parse_fred_csv(poly_text, POLYESTER_SERIES), a.out_dir)
    except FetchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    for f in files:
        print(f"wrote {f}")
    print("Cite the sources in data/SOURCES.txt. Do not commit these files (the IMF cotton series is copyrighted).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

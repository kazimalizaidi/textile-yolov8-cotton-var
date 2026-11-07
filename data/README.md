# Data

Nothing in this folder is redistributed except this file. Download the data yourself:

## Price series (monthly, from FRED)

```bash
python -m textile_risk.fetchdata            # writes three CSVs + SOURCES.txt into ./data
```

| file created | content |
|---|---|
| `cotton_polyester_monthly_fred.csv` | merged file used by `--source csv` (columns `date, cotton_global_usc_per_lb, polyester_ppi_index`) |
| `cotton_global_imf_fred_monthly.csv` | cotton only, from the first month shared with the PPI |
| `polyester_ppi_fred_monthly.csv` | polyester PPI only |
| `SOURCES.txt` | citations and the retrieval date |

* **Cotton**: International Monetary Fund, *Global price of Cotton* [PCOTTINDUSDM], retrieved from FRED, Federal Reserve Bank of
  St. Louis, <https://fred.stlouisfed.org/series/PCOTTINDUSDM>. FRED marks the series "Copyrighted: Citation Required"
  (Copyright (c) International Monetary Fund; terms: <https://www.imf.org/external/terms.htm>), so this repository does **not**
  redistribute it. Users must download the data with the script above to comply with the IMF data terms.
* **Polyester**: U.S. Bureau of Labor Statistics, *Producer Price Index by Commodity: Textile Products and Apparel: Polyester
  Manufactured Fibers* [WPU0315040147], retrieved from FRED, <https://fred.stlouisfed.org/series/WPU0315040147>.
* If the download fails (no network, proxy), open each series page on FRED, press Download > CSV, then run
  `python -m textile_risk.fetchdata --cotton-file path/to/cotton.csv --polyester-file path/to/polyester.csv`.
* FRED data are revised and extended over time; figures in `docs/VALIDATION.md` use data through September 2026
  (retrieved October 2026), so a later download can differ slightly.

Your own data: any CSV with `date` plus a cotton column (`cotton_futures` or `cotton_global_usc_per_lb`, US cents/lb) and a
polyester column (`polyester_spot` or `polyester_ppi_index`). Frequency (daily/weekly/monthly) is detected from the dates;
the minimum number of return observations is 250 / 104 / 120 respectively. Daily cotton futures: `--source yahoo`
(needs `yfinance` and your own polyester CSV).

## Defect images (not included)

Download the figshare dataset "Fabric Defects Object Detection"
(<https://figshare.com/articles/dataset/FD_Dataset_7z/25546465>), follow the licence stated on its page, and extract it to
`data/defects/figshare_fd/`. Native class ids: 0 oil stain, 1 cutting, 2 crack, 3 hole.

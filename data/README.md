# Market data

`--source csv` expects a CSV with exactly these columns (case-insensitive):

| column           | meaning                                   | unit        |
|------------------|-------------------------------------------|-------------|
| `date`           | trading date (ISO `YYYY-MM-DD`)           |             |
| `cotton_futures` | ICE Cotton No.2 front-month settle        | US cents/lb |
| `polyester_spot` | polyester (e.g. PET/PSF) spot price       | USD/kg      |

`sample_template.csv` only shows the header; its row is a zero placeholder and will be
rejected by the loader (prices must be > 0). Replace it with real data.

Sources: cotton -> Yahoo Finance ticker `CT=F` (`--source yahoo`) or ICE/Barchart/Nasdaq
Data Link exports. Polyester has no free exchange-quoted spot feed; use a vendor series
(CCFGroup, ICIS, Bloomberg) or a clearly labelled proxy.

Keep real price files out of git (`data/*.csv` is git-ignored) unless the licence allows it.

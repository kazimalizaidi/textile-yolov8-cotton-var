# 🧵 Textile Quality, Hedging & Risk Platform: Automated Quality Control and Quantitative Risk Management

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat-square&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Scientific_Computing-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-Quality_Engine-6A5ACD?style=flat-square&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Defect_Detection-00A6A6?style=flat-square&logo=ultralytics&logoColor=white)
![Backend](https://img.shields.io/badge/Backend-Engineering-374151?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-2E7D32?style=flat-square)

---

# 📖 Project Overview

An industrial-grade, automated backend pipeline integrating deep learning computer vision, real-time event-driven orchestration, and quantitative market risk hedging. 
It combines three modules:

1. **🔍 Deep Learning Quality Engine** - ingests YOLOv8 fabric-defect detections (holes, oil stains, cracks/cuts), scores each
   batch, and automatically down-values the batch price below a critical quality score.
2. **⚙️ Event-Driven Automation Pipeline** - a Redis-style in-process event broker with worker threads. Three consecutive
   severe defects fire an automatic (mock) request-for-quote to alternative suppliers, and the payload is logged as JSON.
3. **📉 Quantitative VaR Engine** - historical 95% / 99% Value-at-Risk and Expected Shortfall for a Cotton + Polyester book,
   a 20% import-tariff stress test, and a linear-programming hedge optimiser that writes a checksummed JSON config.

Beyond building the system, the project **validates its own results**: it tests the defect model on a held-out shooting session,
documents where the first (optimistic) numbers came from, and states every limitation. See
[`docs/VALIDATION.md`](docs/VALIDATION.md).

The code is object-oriented, fully typed, logged, and covered by an automated test suite (75 tests).

---

## 🏗 Architecture

```text
 images ──> YOLOv8 ──> DetectionIngestor ──> Detection(severity) ──┬──> BatchQualityScorer ──> batch_reports/*.json
                       (validate, clip, NMS,                       │
                        class map, size filter)                    └──> EventBroker (workers)
                                                                         └─ DEFECT_DETECTED ─> ConsecutiveSevereDefectDetector
                                                                               └─ CRITICAL_DEFECT ─> SupplierHedgeOrchestrator
                                                                                     └─> hedge_requests/*.json
 FRED prices ──> HistoricalVaREngine ──> TariffStressTester ──> HedgeOptimizer (LP) ──> hedge_config.json
```

---

## 📂 Datasets

| Dataset | Details |
|---------|---------|
| Defect images | figshare "Fabric Defects Object Detection": 720 labelled photos (540 train / 73 val / 107 test), 781 boxes |
| Classes | oil stain (207), cutting (182), crack (197), hole (195) |
| Shooting sessions | 3 dates: 2023-11-19, 2024-02-10, 2024-02-11 |
| Cotton prices | IMF *Global price of Cotton* via FRED (`PCOTTINDUSDM`), monthly, US cents/lb |
| Polyester prices | BLS *PPI: Polyester Manufactured Fibers* via FRED (`WPU0315040147`), monthly index (Jun 2011 = 100) |
| Price window | Jun 2011 - Sep 2026 (184 months) |

> ⚠️ **Data is not redistributed.** The IMF cotton series is copyrighted and the images are licensed by their authors.
> Download them with the instructions in [`data/README.md`](data/README.md) (`python -m textile_risk.fetchdata` for prices).

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.10+ | Programming |
| 🔢 NumPy | Numerical computing, VaR maths |
| 🐼 Pandas | Time series and CSV handling |
| 📚 SciPy (HiGHS) | Linear-programming hedge optimiser |
| 🎯 Ultralytics YOLOv8 | Defect detection (training / inference, optional) |
| 🧵 `threading` / `queue` | Event broker and worker pool |
| 🖼 Pillow | Dataset inspection tools (optional) |
| ✅ pytest | 75 automated tests |

---

## 📊 Key Results

> Illustrative: assumptions and caveats are in [`docs/VALIDATION.md`](docs/VALIDATION.md). Price figures use data through Sep 2026.

### 🔍 Defect model: random split vs held-out session

| | Random split (107 test images) | Held-out session (125 images, 2023-11-19) |
|---|---|---|
| Precision / Recall | 0.946 / 0.854 | 0.860 / 0.654 |
| mAP50 | **0.918** | **0.732** |
| mAP50-95 | 0.517 | 0.412 |

The random split is **optimistic**: every validation and test photo has a training photo taken within 1 second of it
(neighbouring shots). The held-out session is the realistic estimate. Per class, mAP50 falls from 0.931 to 0.848 for holes, from
0.861 to 0.783 for oil stains and from 0.928 to **0.565 for cracks**.

### 🧪 What the pipeline sees on the held-out session (confidence 0.25)

| Class | True | Found | Missed | False positives |
|---|---|---|---|---|
| Hole | 45 | 35 | 10 | 7 |
| Oil stain | 39 | 28 | 10 | 7 |
| Missing thread (cutting + crack) | 55 | 27 | 28 | 11 |

### 📉 Value-at-Risk (1-month horizon, $2.0M cotton + $1.5M polyester)

| | VaR 95 | VaR 99 | ES 99 |
|---|---|---|---|
| Baseline historical | $165,506 | $282,096 | $410,239 |
| 20% tariff stress | $517,508 | $671,019 | $836,030 |
| Stress after optimal hedge | $204,703 | $250,368 | $271,481 |

The optimiser cuts stressed ES99 by **67.5%** within a $2.0M hedge budget (cotton short $1.8M, polyester short $0.2M).
The 99% estimate rests on only 1.8 tail observations because the data is monthly, and the output file says so.

---

# 📌 Key Findings

- A random train/test split overstated the defect model: mAP50 **0.918** on the random split versus **0.732** on a held-out session.
- Cracks are where the model fails on a new session (recall 0.49). Defect size, shape, the confidence threshold and box tightness
  were each tested and ruled out, which points to domain shift on a narrow, hand-painted-fabric dataset.
- Box sizes for crack and cutting labels vary enormously (0.6% to 59% of the image for cracks), so **area-based severity is
  unreliable** for the merged missing-thread class. An oversized-box filter was built and measured: it changed one prediction.
- The "three consecutive severe defects" trigger is **order-safe under concurrent workers** (sequence-numbered events are
  re-ordered), and a batch fires at most one hedge.
- The hedge optimiser matches a brute-force grid search; the VaR engine matches an independent pandas recomputation, and the
  independent runs on Linux, Windows and Google Colab agree (detection counts, hedge optimum, VaR figures).

---

# 💡 Recommendations & Limitations

- Treat this as a **validated prototype of the method**, not a production quality-control or trading system.
- The dataset contains only defective fabric and has no physical scale: pass the real fabric size per image
  (`--frame-width-m`, `--frame-length-m`) and expect quality scores to scale with it.
- Scoring constants (decay rate, critical score, price elasticity), tariff pass-through values and the supplier downtime cost are
  **assumptions**, not calibrated values.
- Prices are monthly benchmarks, not traded prices: use daily futures data before relying on any 99% figure.
- More diverse defect images (other fabrics, cameras, defect-free samples) are the real lever for the crack problem.

---

## 📂 Repository Structure

```text
├── src/textile_risk/
│   ├── quality.py        # Module 1: ingestion, severity, batch score, price down-valuation
│   ├── events.py         # Module 2: broker, severe-defect detector, supplier hedge
│   ├── risk.py           # Module 3: VaR/ES, tariff stress, LP hedge optimiser, config writer
│   ├── orchestrator.py   # wires the modules together
│   ├── detector.py       # YOLOv8 inference wrapper
│   ├── training.py       # dataset prep, training, evaluation, confusion/review tools
│   ├── datacheck.py      # input checkers (prices, defect dataset, sessions, leakage)
│   ├── fetchdata.py      # downloads the FRED price series
│   ├── common.py, cli.py, __main__.py
├── tests/                # 75 tests: quality, events, risk, training, data checks, end to end
├── docs/VALIDATION.md    # every experiment, number, correction and limitation
├── data/README.md        # how to obtain the data (nothing else is committed)
├── pyproject.toml, requirements*.txt, LICENSE, README.md
```

---

# 🚀 How to Run the Project

1. Clone this repository.

```bash
git clone https://github.com/kazimalizaidi/textile-risk-platform.git
cd textile-risk-platform
```

2. Create an environment and install.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ".[train]" for YOLOv8 training / inference
```

3. Download the price data (not included in the repository, see [`data/README.md`](data/README.md)).

```bash
python -m textile_risk.fetchdata
```

4. Run the tests (75 pass; 4 are skipped until step 3 has been done).

```bash
pytest -q
```

5. Run the risk engine on the real monthly prices (or omit the flags for an offline synthetic demo).

```bash
python -m textile_risk --source csv --prices-csv data/cotton_polyester_monthly_fred.csv --output-dir outputs
```

6. Train and use the defect model (download the figshare images first, see `data/README.md`).

```bash
python -m textile_risk.datacheck defects data/defects/figshare_fd
python -m textile_risk.training prepare --src data/defects/figshare_fd --dst data/defects/prepared
python -m textile_risk.training train   --data data/defects/prepared/data.yaml --epochs 50
python -m textile_risk.training eval    --weights runs/fabric/weights/best.pt --data data/defects/prepared/data.yaml
python -m textile_risk --detector yolo --weights runs/fabric/weights/best.pt \
    --images-dir data/defects/prepared/images/test --frame-width-m 0.30 --frame-length-m 0.20 \
    --source csv --prices-csv data/cotton_polyester_monthly_fred.csv --output-dir outputs
```

`--frame-width-m 0.30 --frame-length-m 0.20` are placeholders: set the real size of fabric in one image.

> Console scripts (`textile-risk`, `textile-risk-train`, `textile-risk-check`, `textile-risk-fetch-data`) are also installed.
> On machines whose policy blocks `.exe` launchers, use `python -m ...` as above.

### 🔬 Validation tools

| Command | What it answers |
|---|---|
| `python -m textile_risk.datacheck sessions data/defects/prepared` | How close in time are test photos to training photos? |
| `python -m textile_risk.datacheck leakage data/defects/prepared` | Are there near-duplicate images across splits? |
| `python -m textile_risk.training prepare ... --holdout-date 2023-11-19` | Hold out a whole shooting session as the test set |
| `python -m textile_risk.training confusion --weights ... --data ...` | Confusion matrices at the pipeline's confidence, native and merged classes |
| `python -m textile_risk.training review --weights ... --class-name crack` | Annotated images and size statistics of missed boxes |
| `python -m textile_risk.training box-limits --data ...` | Per-class box-size limits derived from labels |

### 📁 Outputs

| File | Content |
|---|---|
| `outputs/batch_reports/<id>.json` | score, grade, defect density, price multiplier, value loss |
| `outputs/hedge_requests/hedge_*.json` | full mock RFQ requests and responses, selected supplier |
| `outputs/hedge_config.json` | VaR/ES, stress results, signed hedge weights, contract counts, reliability warnings, SHA-256 |
| `outputs/run_summary.json`, `outputs/logs/pipeline.log` | run metadata, broker stats, rotating log |

---

# 📌 Future Improvements

Potential enhancements include:

- Daily cotton futures (and a daily polyester source) for a statistically meaningful 99% VaR
- Fetching prices through the official FRED API instead of the CSV download
- Defect-free fabric and more fabric types in training and testing
- Length-based severity for linear defects (cracks, cuts) instead of box area
- Calibrating the quality score to a buyer's grading standard
- Replacing the in-process broker with Redis Streams and the mock supplier API with a real client
- Continuous integration (GitHub Actions) running the test suite

---

# 📚 References & Data Sources

- **Defect images:** figshare, *Fabric Defects Object Detection* - <https://figshare.com/articles/dataset/FD_Dataset_7z/25546465>
  (follow the licence on that page; images are not included here).
- **Cotton prices:** International Monetary Fund, *Global price of Cotton* [PCOTTINDUSDM], retrieved from FRED, Federal Reserve
  Bank of St. Louis - <https://fred.stlouisfed.org/series/PCOTTINDUSDM>. Copyright (c) International Monetary Fund; terms of use:
  <https://www.imf.org/external/terms.htm>. *Data sourced from the IMF cotton price series. Users must download the data using
  the provided script (`python -m textile_risk.fetchdata`) to comply with IMF data terms.*
- **Polyester prices:** U.S. Bureau of Labor Statistics, *Producer Price Index by Commodity: Textile Products and Apparel: Polyester
  Manufactured Fibers* [WPU0315040147], retrieved from FRED - <https://fred.stlouisfed.org/series/WPU0315040147>.
- **Method:** Rockafellar, R. T. & Uryasev, S. (2000), *Optimization of conditional value-at-risk*, Journal of Risk 2(3), 21-41
  (the linear-programming formulation of the hedge optimiser).
- **Model:** Ultralytics YOLOv8 - <https://docs.ultralytics.com>.

---

# 📄 License

Released under the [MIT License](LICENSE).

---

# 🎧 Project Soundtrack

*This track was on loop while working on this project*

[bipolar - .diedlonely](https://open.spotify.com/track/1uvn7eP1I7mIKGvpsqDcoW?si=6644ce591da448c5)

**Star this repository if you found it interesting!** ⭐

made with ❤️ by **kazim**

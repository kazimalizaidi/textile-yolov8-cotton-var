# Validation report

What was tested, what was found, what was corrected, and what is still unverified.
Every number below was produced by a command in this repository; reproduction commands are at the end.

## 1. Headline results

| Area | Result | How much to trust it |
|---|---|---|
| Defect model, random split | mAP50 0.918, precision 0.946, recall 0.854 (107 test images) | **Optimistic**: every val/test photo has a train photo within 1 s (see 3.2) |
| Defect model, held-out session | mAP50 0.732 on 3 classes, crack recall 0.49 at conf 0.25 (125 images) | Realistic for a new session of similar fabric; one session only |
| Event pipeline | 3-consecutive-severe trigger is order-safe under concurrent workers, one hedge per batch | Unit-tested (stub detector + real model run on 107 images) |
| VaR engine | Matches an independent pandas recomputation; LP hedge matches a brute-force grid | Maths verified; **inputs are monthly benchmarks**, 99% tail rests on 1.8 observations |
| Oversized-box filter | Measured on the held-out session: removed 1 prediction, no effect on missing-thread | Only the hole limit (3.6% of image) binds; crack/cutting limits are 73% / 100% |
| Quality score constants | Design choices, not calibrated | Needs the buyer's grading standard |

## 2. Data

**Defect images**: figshare "Fabric Defects Object Detection". 720 labelled photos (train 540 / val 73 / test 107),
781 boxes (oil stain 207, cutting 182, crack 197, hole 195), 723 raw photos in `FULLDataset`.
Class ids were confirmed by eye: 0 oil stain, 1 cutting, 2 crack, 3 hole. The labelled set is re-exported from the raw
photos (one export per photo, 720 unique originals; files with " (2)" in the raw name appear renamed to `-2-` on export, seen on one
file and inferred from counts for the other 39; 3 raw photos have no labelled export). Photos come from three dates: 2023-11-19 (125 labelled), 2024-02-10 (180), 2024-02-11 (415).
Within the November session the model's predictions arrive in runs by class (oil stains, then cracks, then holes),
which suggests the photos were taken class by class; this is an inference from prediction order, not from the labels.

**Prices**: FRED `PCOTTINDUSDM` (IMF global cotton, US cents/lb, Cotlook "A" based) and FRED `WPU0315040147`
(BLS polyester PPI, Jun 2011 = 100). 184 monthly rows, 2011-06 to 2026-09 (series pages updated 22 Oct 2026).
**The data are not redistributed** (FRED marks the IMF cotton series "Copyrighted: Citation Required"); users fetch them with
`python -m textile_risk.fetchdata`. The figures in this report were computed on a hand-transcribed copy of the FRED tables:
cotton 2021-2025 annual means matched FRED's published aggregates to five decimals, 2011-2020 values were not independently
verified. The downloader reproduces that copy exactly in an offline round-trip test; it has **not** been run against the live FRED
endpoint by the author. Later downloads include newer months and revisions, so figures can differ slightly.
Annualised volatility 17.2% (cotton) and 6.7% (polyester); return correlation 0.245; the polyester PPI is unchanged in 24 of 183
months (13%), i.e. smoothed. **Neither series is a traded price.**

## 3. Module 1: defect model

### 3.1 Random split (YOLOv8, 50 epochs)
| class | mAP50 | mAP50-95 |
|---|---|---|
| cutting | 0.950 | 0.601 |
| hole | 0.931 | 0.463 |
| crack | 0.928 | 0.562 |
| oil stain | 0.861 | 0.443 |
| **all** | **0.918** | **0.517** |

Best validation epoch 46 (mAP50 0.929). Validation loss levelled off and did not rise (no overfitting signature).
The jump after epoch 40 coincides with the default mosaic-augmentation shutoff for the last 10 epochs (a plausible cause; not tested).

### 3.2 Leakage checks
- Byte-identical images across splits: 0.
- Perceptual near-duplicates (64-bit dHash incl. flips/rotations, <= 6 bits): 0/73 val, 0/107 test.
- **Temporal neighbours: 73/73 val and 107/107 test photos have a train photo within 1 second** (median gap 1 s, same as
  train-to-train). The random split therefore tests on near-neighbours of training images, so 0.918 is an upper bound.

### 3.3 Held-out session (train on Feb 2024, test on all 125 photos of 2023-11-19)
| | random split | Nov hold-out |
|---|---|---|
| precision / recall | 0.946 / 0.854 | 0.860 / 0.654 |
| mAP50 / mAP50-95 | 0.918 / 0.517 | 0.732 / 0.412 |

Nov test boxes: oil stain 39, cutting 0, crack 55, hole 45 (cutting appears only in the February sessions, so the overall
numbers average three classes; like-for-like on those three classes the random split scores 0.907, a drop of about 0.17).

| class | mAP50 random | mAP50 Nov | change |
|---|---|---|---|
| hole | 0.931 | 0.848 | -0.08 |
| oil stain | 0.861 | 0.783 | -0.08 |
| crack | 0.928 | 0.565 | -0.36 |

### 3.4 What the pipeline would see on the held-out session (conf 0.25, IoU 0.5, one-to-one matching)
| class | true | detected correctly | missed | false positives | recall | precision |
|---|---|---|---|---|---|---|
| hole | 45 | 35 | 10 | 7 | 0.778 | 0.814 |
| oil stain | 39 | 28 | 10 | 7 | 0.718 | 0.800 |
| missing thread (cutting+crack) | 55 | 27 | 28 | 11 | 0.491 | 0.711 |

Net count error across all classes about -16% (116 predicted vs 139 true), so batch defect density would be understated on
a new session. Sensitivity: IoU 0.3 recovers 2 more cracks (recall 0.527); confidence 0.1 recovers 3 more cracks but
adds 21 false positives (missing-thread precision 0.484).

### 3.5 Why cracks are missed (investigation, in order)
1. *Crack confused with cutting?* No: 0 of 55 cracks were predicted as cutting (1 at IoU 0.3).
2. *Size or thin shape?* No: found vs missed cracks have the same medians (short side 101 vs 93 px, long side 251 vs 297 px,
   area 5.7% vs 5.3% of the image, aspect 2.3 vs 2.4). Individual cases do vary (a 30x80 px tear and a 400x36 px slit were both missed).
3. *Loose boxes?* Only 2 of 28 recovered at IoU 0.3.
4. *Confidence threshold?* No: see 3.4.
5. *Where do the 28 problem cracks end up?* 14 no prediction anywhere in the image, 8 nearest prediction overlap < 0.1,
   4 overlap 0.1-0.3, 2 overlap 0.3-0.5. So over 40% are **mislocalised**, not blank (an earlier working hypothesis that most
   were blanks was wrong; a seam-edge tear was covered by two oversized boxes at 0.42 and 0.27 confidence).
6. *Label noise?* In the three images inspected the labelled box sits on a real visible defect (3 of 28 only).
7. *Original model (trained on 88 of these 125 photos) on the same images:* crack recall 0.891, oil stain 1.00, hole 0.867.
   Contaminated by design (about 70% were training images); it shows labels are largely learnable and that seeing neighbouring
   November photos lifts crack recall from 0.49 to 0.89.

**Conclusion:** domain shift on a narrow dataset (a few hand-painted fabric pieces from one photographer), not resolution,
thresholds or box tightness. More diverse training data is the real lever.

### 3.6 Oversized-box filter: built, measured, and found not to help
`textile-risk-train box-limits` derives a per-class maximum box area from the **train and val labels only**
(limit = 1.25 x largest labelled box); the ingestor rejects larger boxes before NMS (counted as `oversized_box`) and
`confusion --box-limits` measures the effect on any model. The motivation was oversized low-confidence boxes inflating severity
(severity grows with box area) and triggering false "3 consecutive severe" hedges.

Labelled box area, % of image (train+val of the held-out-session split, February photos):

| class | n | median | p90 | p99 | max | limit |
|---|---|---|---|---|---|---|
| hole | 150 | 0.7 | 1.4 | 2.2 | 2.9 | 3.6 |
| oil stain | 168 | 7.2 | 14.7 | 25.1 | 50.2 | 62.7 |
| crack | 142 | 9.4 | 32.3 | 56.0 | 58.7 | 73.4 |
| cutting | 182 | 17.0 | 49.4 | 80.3 | 96.2 | 100 |

Because cutting and crack are merged into MISSING_THREAD and the larger limit wins, that class's limit is 100% of the image,
so the filter cannot reject anything there. Measured on the 125 held-out November photos (conf 0.25, IoU 0.5):

| merged class | found | missed | false positives | precision |
|---|---|---|---|---|
| hole | 35 -> 35 | 10 -> 10 | 7 -> 7 | 0.814 -> 0.833 |
| oil stain | 28 -> 29 | 10 -> 10 | 7 -> 6 | 0.800 -> 0.829 |
| missing thread | 27 -> 27 | 28 -> 28 | 11 -> 11 | 0.711 -> 0.711 |

The filter removed exactly one prediction (an oversized "hole" that had been matched to an oil stain). It does not address the
crack problem: the oversized boxes seen on the seam-edge tear (about 15% and 35% of the image) are well inside the range of
boxes that annotators drew for cracks and cuttings. **Finding:** labelled boxes for linear defects vary enormously in size
(crack 0.6% to 58.7% of the image, cutting up to 96%), which suggests an inconsistent box convention (the tear only vs the damaged region).
Area-based severity is therefore unreliable for MISSING_THREAD. Additionally, the November test cracks have a median area of
about 5.5% (found 5.7%, missed 5.3%) against 9.4% for the February training cracks, a shift in box size between sessions.
This is a contributing factor, not the explanation: at IoU 0.3 only 2 more cracks are recovered.
The filter is kept as an optional, data-derived guard; in practice only the hole limit is informative.

## 4. Module 2: event pipeline
Automated tests (75; 4 of them skip until the price data is downloaded) cover: fan-out delivery, retry then dead-letter, three consecutive severe defects firing exactly once
even when events arrive out of order, streak reset on a non-severe defect, per-batch isolation, one hedge per batch
(later triggers suppressed), deterministic mock supplier responses, and atomic JSON writes.
A run on 107 real test images produced 127 detections, 5 critical events and 1 hedge request; the same run on Windows and on Colab gave
identical counts (127 detections: 36 hole / 23 oil stain / 68 missing thread; 30 severe; 5 critical events; 1 hedge) and weighted density
agreeing to 9e-8. Reading the hedge file exposed a bug: every mock quote id was `Q-000000000000` (fixed, with a regression test).
That run used placeholder frame sizes (0.30 x 0.20 m); its quality score (3.14, grade D) is an artefact of
every test image containing a defect plus the assumed scale, **not a finding**.

## 5. Module 3: VaR engine
- Historical VaR/ES at 95% and 99% on P&L = sum N_i (exp(r_i) - 1); cross-checked against an independent pandas recomputation on every run (and in a test on the real file).
- Hedge optimiser: Rockafellar-Uryasev LP. On the synthetic series its objective ($196,615.54) matches a brute-force grid optimum
  within $11 (estimator difference); the same value was reproduced on a separate Windows machine.
- The final Colab run (all-session model + real monthly prices) reproduced every figure below to the cent.
- Real monthly run (default book: $2.0M cotton, $1.5M polyester, 20% tariff shock, vol x1.5):

| | VaR 95 | VaR 99 | ES 95 | ES 99 |
|---|---|---|---|---|
| baseline, 1 month | $165,506 | $282,096 | $264,373 | $410,239 |
| tariff stress | $517,508 | $671,019 | $647,214 | $836,030 |
| stress after hedge | $204,703 | $250,368 | $228,201 | $271,481 |

Stressed ES99 falls 67.5% with a $2.0M hedge budget: cotton shorted $1.8M (ratio 0.90, 38 contracts), polyester $0.2M.
Instantaneous tariff-shock P&L: -$291,555. The output file states its own limits: monthly horizon, 99% tail on 1.8
observations, benchmark/index inputs.

## 6. Known limitations and assumptions
- Daily data paths (`--source yahoo`) are written but **untested**; no free daily polyester series was found.
- The FRED downloader (`textile_risk.fetchdata`) is tested offline only (parsing, validation, merge, error handling, round trip).
- Tariff parameters (pass-through 0.35 / 0.60, direction, 1.5x vol) are placeholders. The polyester hedge assumes no basis risk.
- Quality score constants (kappa 0.35, critical score 70, price elasticity 2, severity weights) and the supplier downtime cost ($12,000/day) are design choices.
- The dataset has no physical scale: `--frame-width-m/--frame-length-m` must be supplied; scores scale with them.
- The dataset contains only defective fabric, so false-alarm behaviour on clean fabric is untested.
- Cutting and crack are both mapped to MISSING_THREAD (a judgement call; configurable with `--class-map`).
- Box sizes for crack/cutting labels are highly variable (3.6), so area-based severity is unreliable for MISSING_THREAD.
- The supplier API is a deterministic mock and the broker is in-process.

## 7. Reproduce
```bash
python -m textile_risk.fetchdata        # price data is not in the repository
pytest -q
textile-risk-check prices data/cotton_polyester_monthly_fred.csv
textile-risk-check leakage data/defects/prepared --sheet leak_pairs.png
textile-risk-check sessions data/defects/prepared
textile-risk-train prepare --src data/defects/figshare_fd --dst data/defects/prepared_nov --holdout-date 2023-11-19
textile-risk-train train --data data/defects/prepared_nov/data.yaml --epochs 50 --name fabric_nov
textile-risk-train eval --weights runs/fabric_nov/weights/best.pt --data data/defects/prepared_nov/data.yaml
textile-risk-train confusion --weights runs/fabric_nov/weights/best.pt --data data/defects/prepared_nov/data.yaml
textile-risk-train review --weights runs/fabric_nov/weights/best.pt --data data/defects/prepared_nov/data.yaml --class-name crack
textile-risk-train box-limits --data data/defects/prepared_nov/data.yaml --out box_limits_nov.json
textile-risk-train confusion --weights runs/fabric_nov/weights/best.pt --data data/defects/prepared_nov/data.yaml --box-limits box_limits_nov.json
textile-risk --source csv --prices-csv data/cotton_polyester_monthly_fred.csv --output-dir outputs
```

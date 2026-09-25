# Research after the first submission

The experiments behind section 8 of the EDA website. None of them beat the submitted model by the +0.005 CV AUC margin we set, so the submitted model (`../escalation_model.ipynb`) is unchanged.

Every experiment uses the same evaluation: 5-fold stratified CV repeated with 5 seeds, reporting mean AUC across seeds.

| Script | What it tests | CV AUC |
|---|---|---|
| `01_baseline.py` | submitted model | 0.6470 |
| `02_windowed.py` | + windowed per-type amount quantiles | 0.6475 |
| `03_behaviour.py` | + floor, burst, weekly rhythm, transitions | 0.6477 |
| `04_residual.py` | + customer amount level | 0.6478 |
| `05_histogram.py` | + full amount histograms | 0.6460 |
| `06_tuning.py` | LightGBM / LogReg settings | 0.6431–0.6473 |
| `07_ensemble.py` | ensemble of variants | 0.6482 |
| `08_topk.py` | top-30 features chosen inside each fold | 0.6485 |
| `09_structure_checks.py` | IDs, file order, date effects | AUC 0.49–0.51 |
| `10_site_numbers.py` | numbers quoted on the website | — |

## Run

Put the competition files in the repo root (next to the notebook), or point `WES_DATA` at their folder, then run the scripts in order from this folder:

```bash
cd research
python 01_baseline.py        # later scripts reuse its cached results
python 02_windowed.py
...
```

Features and predictions are cached in `research/cache/` (not committed).

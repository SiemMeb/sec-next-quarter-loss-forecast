# Next-Quarter Loss Forecast from SEC Filings

![cover](assets/cover.jpg)

**Will a U.S. public company report a net loss next quarter?** This project builds a leakage-safe forecast from the financial statements companies file with the SEC (Forms 10-K and 10-Q). It then checks the forecast against the simplest rule that could replace it: "a company losing money now will lose money next quarter".

[![ci](https://github.com/SiemMeb/sec-next-quarter-loss-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/SiemMeb/sec-next-quarter-loss-forecast/actions/workflows/ci.yml)
[![Kaggle notebook](https://img.shields.io/badge/Kaggle-notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/siemmebrahtu/next-quarter-loss-forecast-from-sec-filings)
[![Kaggle dataset](https://img.shields.io/badge/Kaggle-dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/siemmebrahtu/sec-10k-10q-wide-v3)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-HistGradientBoosting-F7931E?logo=scikitlearn&logoColor=white)
[![License: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
![Data: CC0](https://img.shields.io/badge/data-CC0-lightgrey)

---

## Results at a glance

![results](assets/results.svg)

| Measure (test period: label quarters 2025-Q4 onward) | Result |
|---|---|
| Gradient boosting ROC-AUC | **0.924** (95% CI 0.918 to 0.929, company-clustered bootstrap) |
| Gradient boosting PR-AUC | **0.911**, compared with 0.755 for persistence |
| Lift over persistence, filing-date protocol | **+0.099** ROC-AUC (CI +0.094 to +0.106) |
| Rolling-origin evaluation | beats persistence in **7 of 7** origins (mean lift +0.102) |
| Walk-forward, filing-date protocol | beats persistence in **8 of 8** origins (mean lift +0.107) |
| Transitions only (profitable now, loss next) | ROC-AUC 0.847; PR-AUC 0.597, which is 3.5 times the 17% event rate |
| Companies never seen in training | ROC-AUC 0.890 |
| Threshold chosen on validation (0.424) | test precision 0.823, recall 0.872 |
| Company-grouped cross-validation (preparation refitted inside each fold) | boosting 0.933 ± 0.003, logistic 0.915 ± 0.004 |
| **Single-use final test** | **Registered, pending.** 5,770 company forecasts and the frozen model are saved before their outcomes exist |

The model has **not been tuned**. Its settings were fixed in advance, and every later choice (features, cheaper model, thresholds) was made on the validation period.

## The question

| Target | Definition | Status |
|---|---|---|
| **S1** (main) | Next quarter's standalone net income is below zero | Main result |
| **S2** (exploratory) | Total equity turns negative next quarter | Promising screen, not established (see below) |
| **U1** | k-means "financial archetypes" (k = 7) used as an extra feature | Adds nothing on validation (−0.0003 ROC-AUC); reported as a negative result |

## Why the evaluation can be trusted

Most of the work goes into ruling out the usual ways a financial forecast leaks the future.

- **Time split by label quarter.** Training labels end on 2024-12-31, validation runs to 2025-09-30, and test comes after that.
- **Filing-date protocol.** A label counts as known only once the next filing has actually been submitted (`label_known_date`). Under the plain label-quarter split, 17% of training labels were not yet public, and the filing-date version gives the same answer (test 0.925, validation 0.933).
- **Leakage-prone fields removed.** For example, the amendment flag is excluded, because whether a filing gets amended is only known afterwards.
- **Clip limits fitted on training rows only**, including inside every cross-validation fold.
- **Strong baselines.** The model has to beat persistence and a history table, not a coin flip.
- **A single-use final test, registered in advance** (next section).

## A final test that cannot have been tuned on

The test period above was looked at more than once while the notebook was built, so it is not a pristine hold-out. Cell 21b fixes this.

1. **Freeze.** The model, its clip limits, its feature list and its validation-chosen threshold are saved to `frozen_s1/`, together with a fingerprint of its validation predictions.
2. **Register.** Each company's latest filing in the data gets a forecast for its next quarter. That outcome is in none of the files used, and the 5,770 forecasts are saved before any of them is known.
3. **Pre-specify.** `final_test_plan.json` fixes everything in advance:
   - the test set
   - the primary measure: ROC-AUC lift over persistence with a company-clustered 95% interval
   - the success rule: the whole interval must lie above zero
   - the secondary measures
4. **Run once.** The analysis reads quarter files only up to `ANALYSIS_LAST_QUARTER = "2026q2"`. Once newer DERA quarters are added to the dataset, they are read **only** by Cell 21b, so no earlier number can move. The result is reported whatever it is.

The final-test code path is exercised in CI by holding back the last quarter of a synthetic panel (`FINAL_TEST=1`).

## From scores to decisions

- **Class balance (Cell 20b).** The three tasks have very different event rates:

  | Task | Test event rate | Model PR-AUC | Times the no-skill value |
  |---|---|---|---|
  | S1 | 47% (close to balanced) | 0.911 | 1.9 |
  | S1 transition | 17% | 0.597 | 3.5 |
  | S2 | 1.6% (rare) | 0.323 | 19.7 |

  PR-AUC is always read against its no-skill value, which equals the event rate.
- **Error costs (Cell 20b).** Let *c* be the cost of a missed loss divided by the cost of a false alarm. For each *c* from 0.5 to 10, a threshold is chosen on validation and applied unchanged to test.
  - For **S1**, the model's expected cost is 14% to 46% lower than the best simple policy at every cost ratio. The simple policies are persistence, flag nothing and flag everything.
  - For **S2**, the burn-rate rule is slightly cheaper when misses are cheap (*c* = 1 to 2). The model is cheaper only when misses are costly (*c* = 5 to 10): 8% to 10% lower cost. This is another reason S2 stays exploratory.
- **Screening list (Cell 21c).**
  - Companies that are losing money now are not the interesting part, because persistence already flags them. The useful list is **profitable now, at high risk of a loss next quarter**.
  - In validation, the top 25 of such a list turned to a loss **96%** of the time, the top 100 **81%**, against a base rate of 16%.
  - Fund-like filers and SPACs are marked rather than hidden.
  - The list is saved as `screening_list.csv`, and `app/streamlit_app.py` browses it.

## What drives the forecast

Cell 11b checks whether the ranking of drivers is stable, rather than reading one importance chart.

- **Across company folds:** permutation importance is re-measured in 5 company-grouped folds. Rank agreement between folds has a mean Spearman correlation of 0.89, and the lowest is 0.85.
- **Across time:** permutation importance is re-measured in each of the 5 validation and test quarters.
- **Against SHAP:** the SHAP ranking agrees with the permutation ranking (Spearman 0.85, 8 of the top 10 shared).

**Stable drivers**, which stay in the top 10 in every fold and every quarter:

- this quarter's profit relative to assets
- operating margin
- profit one, two and three quarters ago (seasonality)
- retained earnings relative to assets
- company size

All of them push in the expected direction: more profit, more retained earnings and larger size lower the risk.

One detail matters for anyone reusing SHAP with scikit-learn's `HistGradientBoostingClassifier` with categorical features. The fast tree explainer **failed the additivity check** in this setup: its values missed the model's own output by up to 4.9 log-odds. The notebook therefore checks additivity first and falls back to the model-agnostic permutation explainer, whose values add up exactly. These are associations the model relies on, not causal effects.

## S2: an exploratory negative-equity screen

S2 ranks companies well:

- ROC-AUC is 0.959, and the ROC-AUC lift over the burn-rate rule is +0.121 [+0.082, +0.164].
- The top 10% of scores capture 88.7% of events.

However, there are only 141 test events. The PR-AUC lift over the best rule is +0.060 [−0.011, +0.131], and that interval **crosses zero**. S2 is therefore reported as a promising exploratory screen, not a proven improvement.

## Where it works less well

- **Fund-like filers** (BDCs, credit and income funds) score ROC-AUC 0.613, against 0.927 for everyone else. Their losses follow market valuations that filings do not foreshadow.
- **Turning points** are harder than steady states (0.847 on transitions).
- A cheaper **top-8-feature boosting model** keeps 96% of the lift on validation (95% on test). A top-8 logistic regression keeps only 46%.

## Repository layout

```
.
├── notebooks/next-quarter-loss-forecast.ipynb   # the full analysis (Parts 1 to 3, final test, screening list, model card)
├── src/lossforecast/                            # the S1 pipeline as tested modules
│   ├── data.py          # quarter files, analysis window, strict column check
│   ├── labels.py        # standalone quarter by period chain, next-quarter label
│   ├── features.py      # lags, the 30 features, row filter, time split
│   ├── models.py        # train-only clipping, gradient boosting, logistic regression
│   ├── evaluation.py    # metrics, company-clustered bootstrap, thresholds, error costs
│   ├── screening.py     # registered forecasts, final-test labels, screening list
│   └── pipeline.py      # end-to-end command line run
├── tests/
│   ├── test_labels.py, test_features.py, test_models_evaluation.py   # unit tests, incl. a no-look-ahead test
│   ├── test_parity.py   # runs the notebook's own cells and checks the modules reproduce them exactly
│   ├── smoke_test.py    # runs every notebook cell on a synthetic panel (FINAL_TEST=1: final-test branch)
│   └── synthetic.py     # the synthetic panel (same schema, no download)
├── app/streamlit_app.py # screening-list viewer
├── docs/DATA_CARD.md, docs/reports/*.pdf
├── .github/workflows/ci.yml                     # tests + both smoke tests on every push
├── requirements.txt     # exact versions of the Kaggle run (requirements-dev.txt for CI, requirements-app.txt for the app)
└── pyproject.toml
```

## How to run

**On Kaggle (easiest).** Open the [notebook](https://www.kaggle.com/code/siemmebrahtu/next-quarter-loss-forecast-from-sec-filings) and click *Copy & Edit*. The dataset is already attached.

**Locally.**

```bash
git clone https://github.com/SiemMeb/sec-next-quarter-loss-forecast.git
cd sec-next-quarter-loss-forecast
pip install -r requirements.txt && pip install -e . --no-deps
kaggle datasets download -d siemmebrahtu/sec-10k-10q-wide-v3 -p data --unzip

python -m lossforecast.pipeline --data data --out results     # headline table + screening list
jupyter lab notebooks/next-quarter-loss-forecast.ipynb        # set DATA_DIR = "data" in Cell 1
```

**Tests (no download needed).**

```bash
pip install -r requirements-dev.txt && pip install -e . --no-deps
pytest -q                                  # unit tests and notebook parity
python tests/smoke_test.py                 # every notebook cell on a synthetic panel
FINAL_TEST=1 python tests/smoke_test.py    # the same, with the final-test branch
```

**Screening app.**

```bash
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py         # upload screening_list.csv from the notebook's Output tab
```

The notebook stays self-contained so that it runs on Kaggle without this package. `test_parity.py` keeps the two in step: it executes the notebook's own cells and requires the modules to reproduce the labels, the features, the split and the model's test predictions exactly.

## Data

The data comes from the **SEC Financial Statement Data Sets** (Division of Economic and Risk Analysis), reshaped into one row per canonical filing:

- 81,472 filings from more than 6,000 companies
- 14 quarterly releases, 2023-Q1 to 2026-Q2
- 1,730 columns, including standalone-quarter (`_qtr`) and trailing-twelve-month (`_ttm`) measures

See [docs/DATA_CARD.md](docs/DATA_CARD.md) and the [Kaggle dataset](https://www.kaggle.com/datasets/siemmebrahtu/sec-10k-10q-wide-v3).

## Limitations

- **Short history.** There are about two years of training labels, so seasonality is learned from at most two cycles.
- **Values are as filed.** Scale errors are handled with ratios and clipping, not corrected.
- **The final test is registered but not yet run.** It needs the next DERA quarters.
- **S2 is exploratory.**
- **Explanations describe the model, not causes.** Correlated features share credit under both permutation importance and SHAP.
- **Scope.** The model card (Cell 22) lists the intended uses and the uses the model is not meant for.

## For forks and extensions

- Add the next DERA quarters to the dataset and re-run the notebook unchanged. This runs the single-use final test.
- Use separate models or targets for **fund-like filers**.
- Collect more S2 events (a longer history) before drawing conclusions about negative equity.
- Add **text features** from MD&A or going-concern language, tested with the ablation in Cell 12c.

## License

The code is under the [MIT License](LICENSE). The data is U.S. government data in the public domain, and the derived dataset is released under CC0.

*This project is for research and education only. It is not investment advice.*

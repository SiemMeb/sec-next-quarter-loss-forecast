# Data card: SEC 10-K and 10-Q wide panel (ETL v3.2)

Copy of the dataset description published on Kaggle: https://www.kaggle.com/datasets/siemmebrahtu/sec-10k-10q-wide-v3

An analysis-ready panel of the financial statements that U.S. public companies file with the SEC on Forms 10-K and 10-Q, built from the SEC's **Financial Statement Data Sets** (Division of Economic and Risk Analysis, DERA). Each row is one canonical filing (amendments resolved); each column is one standard us-gaap XBRL tag, a derived measure or filing metadata.

**At a glance**
- **81,472 filings** from more than 6,000 companies, submitted from 2023Q1 to 2026Q2 (14 quarterly files)
- **1,730 columns**: balance sheet, income statement, cash flow and per-share values, plus filing metadata
- **Standalone quarters (`_qtr`) and trailing twelve months (`_ttm`)** for the main income-statement measures, so rows can be compared across quarters
- **1,662 superseded filings** (earlier versions of amended filings) kept in a separate file
- Parquet format, CC0 public domain

### Files

| File | Contents |
|---|---|
| `2023q1.parquet` to `2026q2.parquet` | Canonical filings, one file per DERA release quarter (the quarter in which the filing was submitted). Read them together with the pattern `20??q?.parquet`. |
| `superseded_filings.parquet` | Earlier versions of amended filings, with the same columns plus `superseded_by` (the `adsh` of the filing that replaced each one). It is not part of the panel, so leave it out when reading the quarter files. |

### How to read the columns

**Income statement (read this first)**
- Unsuffixed flow columns (`Revenues`, `NetIncomeLoss`, `OperatingIncomeLoss`, ...) are **cumulative from the start of the fiscal year**. `dur_qtrs` (1, 2, 3 or 4) gives the number of quarters they cover: Q1 = 3 months, Q2 = 6, Q3 = 9, 10-K = 12. They are not comparable across rows with different `dur_qtrs`.
- `<measure>_qtr` is the **standalone quarter**: the reported three-month figure when the filing has one, otherwise this cumulative value minus the previous cumulative value of the same company. The previous row is found by period order: it must end 75 to 110 days earlier and cover exactly one quarter fewer. The value is blank when that previous row is missing. `NetIncomeLoss_qtr_src` says `reported` or `derived`.
- `<measure>_ttm` is the **trailing twelve months**: the annual figure on 10-K rows, otherwise the sum of four consecutive standalone quarters.
- Measures with `_qtr` and `_ttm`: `Revenues`, `CostOfRevenue`, `GrossProfit`, `OperatingIncomeLoss`, `NetIncomeLoss`, `InterestExpense`. `EarningsPerShareBasic_qtr` holds reported quarters only, and there is no TTM for EPS.

**Cash flow:** `<tag>_ytd` is the cumulative value as reported, and `<tag>_qtr` is the derived standalone quarter, for `NetCashProvidedByUsedInOperatingActivities`, `...InvestingActivities` and `...FinancingActivities`.

**Balance sheet:** values are snapshots at the period end (`Assets`, `Liabilities`, `EquityTotal`, `CashAndCashEquivalentsAtCarryingValue`, ...).

**Metadata and flags**

| Column | Meaning |
|---|---|
| `adsh` | SEC accession number of the filing (unique key) |
| `cik` | SEC company identifier |
| `name`, `countryba` | Company name and business-address country |
| `sic`, `sic_name`, `sic_division` | Standard Industrial Classification code, name (common codes) and division (A to J) |
| `form`, `fy`, `fp` | Form type (10-K, 10-Q, amendments), fiscal year label, fiscal period (Q1, Q2, Q3, FY) |
| `period` | Period end date of the statements |
| `filed`, `filed_date` | Filing date (raw integer and date) |
| `days_to_file` | Days from period end to filing |
| `dur_qtrs` | Quarters covered by the cumulative flow columns (1 to 4) |
| `quarter_source` | DERA release quarter the filing came from (same as the file name) |
| `is_amended`, `n_versions` | Whether the company-period was filed more than once, and how many versions there were |
| `is_spac` | SIC code 6770 (blank-check companies) |
| `is_shell` | Heuristic: assets under USD 1M and no revenue |
| `has_negative_nonneg_tag` | A tag that can never be negative (assets, cash, receivables, ...) has a negative value |
| `liabilities_derived`, `liabilities_derivation_conflict` | `Liabilities` was filled from Liabilities and equity minus equity, or the fill was refused because it would be negative |
| `<family>_tag_used` | Which synonym tag supplied `Revenues`, `CostOfRevenue`, `NetIncomeLoss` or `EquityTotal` |

### How it was built
1. Read `sub.txt` and `num.txt` from each DERA quarterly release; keep forms starting with 10-K or 10-Q, including amendments.
2. Keep facts for the current reporting period only, consolidated totals only (no segments or co-registrants), units USD, USD per share and shares, standard us-gaap tags only, and tags used by at least 20 filings in the quarter.
3. Pivot to one row per filing and one column per tag.
4. Coalesce common synonym tags for `Revenues`, `CostOfRevenue`, `NetIncomeLoss` and `EquityTotal`.
5. Fill a missing `Liabilities` from the accounting identity only when the result is not negative.
6. Resolve amendments: keep the latest filed version of each (`cik`, `period`, `fp`); move earlier versions to `superseded_filings.parquet`.
7. Derive standalone quarters and trailing twelve months, then add the metadata and flags.
8. Validate, then write one parquet file per release quarter.

### Quality checks on this version
- `(cik, period, fp)` and `adsh` are unique; `dur_qtrs` is always 1 to 4.
- Assets equal liabilities plus equity within 1% for at least 99% of filings.
- The four standalone quarters add up to the annual `NetIncomeLoss` for at least 90% of the 14,748 complete fiscal years.
- Coverage: `NetIncomeLoss_qtr` 92.5% (59,601 reported, 15,783 derived), `NetIncomeLoss_ttm` 74.0%, `Revenues` 73.5%, `CostOfRevenue` 44.9%; `Liabilities` missing for 1.0%.
- An independent rebuild of the derived quarters in the analysis notebook disagrees with the published values in 0.00% of cases.

### Things to know before modelling
- **Values are as filed.** Some tags contain scale errors (for example a few EPS values in the hundreds of thousands per share). Prefer ratios to assets over per-share values, and check `has_negative_nonneg_tag` (6 rows).
- **The canonical key includes `fp`.** A handful of company-period pairs (9 in this version) appear twice with different `fp`. If you need exactly one row per company and quarter, keep the latest filed.
- **Filing delays vary widely.** `days_to_file` includes negative values and catch-up filings years late; filtering to 0 to 365 days removes about 650 rows.
- **SPACs** often report negative equity because redeemable shares sit outside equity.
- **Memory:** with 1,730 columns, read only the columns you need (`pd.read_parquet(path, columns=[...])`).

### Starter code
```python
import glob, pandas as pd

files = sorted(glob.glob("/kaggle/input/sec-10k-10q-wide-v3/20??q?.parquet"))  # quarter files only
cols = ["adsh", "cik", "name", "form", "fp", "period", "dur_qtrs", "Assets", "NetIncomeLoss_qtr", "NetIncomeLoss_ttm"]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
```
The mount path can differ between notebooks (for example `/kaggle/input/datasets/siemmebrahtu/sec-10k-10q-wide-v3`); check it in the notebook's Input panel.

### Source and license
Source: U.S. SEC, Division of Economic and Risk Analysis, Financial Statement Data Sets (https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets). U.S. government data in the public domain; this derived dataset is released under CC0.

"""Reading the quarter files of the `sec-10k-10q-wide-v3` dataset (notebook Cell 2)."""
import glob
import os

import pandas as pd

REQUIRED = ["adsh", "cik", "name", "sic_division", "form", "fp", "period", "dur_qtrs", "days_to_file", "is_spac", "is_shell", "is_amended",
            "has_negative_nonneg_tag", "NetIncomeLoss_qtr_src",
            "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent", "EquityTotal", "RetainedEarningsAccumulatedDeficit",
            "CashAndCashEquivalentsAtCarryingValue",
            "NetIncomeLoss", "NetIncomeLoss_qtr", "NetIncomeLoss_ttm", "Revenues_qtr", "OperatingIncomeLoss_qtr",
            "NetCashProvidedByUsedInOperatingActivities_qtr", "NetCashProvidedByUsedInFinancingActivities_qtr"]
OPTIONAL = ["GrossProfit_qtr"]
FLAG_INPUTS = ("is_spac", "is_shell", "is_amended", "has_negative_nonneg_tag")


def quarter_files(data_dir, last_quarter):
    """Split the quarter files into the analysis window (up to `last_quarter`) and the newer files kept for the final test."""
    files = sorted(glob.glob(os.path.join(data_dir, "20[0-9][0-9]q[1-4].parquet")))      # never superseded_filings.parquet
    q = lambda f: os.path.basename(f)[:6]
    return [f for f in files if q(f) <= last_quarter], [f for f in files if q(f) > last_quarter]


def normalise(raw):
    """Parse the period and turn the True/False/missing flags into 1.0/0.0/NaN."""
    out = raw.copy()
    out["period"] = pd.to_datetime(out["period"])
    for c in FLAG_INPUTS:
        out[c] = out[c].astype("boolean").astype("float")
    return out


def load_quarters(files):
    """Read only the columns the S1 model needs, with a strict check by column name."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    if not files:
        raise FileNotFoundError("no quarter files given")
    unified = pa.unify_schemas([pq.read_schema(p) for p in files], promote_options="permissive")
    missing = [c for c in REQUIRED if c not in unified.names]
    if missing:
        raise KeyError(f"columns not in the dataset: {missing} (an old dataset version: ETL v3.2 is required)")
    cols = REQUIRED + [c for c in OPTIONAL if c in unified.names]
    return normalise(ds.dataset(files, format="parquet", schema=unified).to_table(columns=cols).to_pandas())

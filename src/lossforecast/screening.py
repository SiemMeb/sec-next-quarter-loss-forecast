"""Registered forecasts, the single-use final test and the screening list (notebook Cells 21b and 21c)."""
import numpy as np
import pandas as pd

from .config import Settings

FUND_PATTERN = (r"PRIVATE CREDIT|DIRECT LENDING|SPECIALTY LENDING|LENDING CORP|LENDING FUND|LENDING PARTNERS|\bBDC\b|CAPITAL TRUST|"
                r"CREDIT FUND|CREDIT INCOME|CREDIT OPPORTUNITIES|CREDIT STRATEGIES|INCOME CORP|SENIOR LOAN")
FUND_EXCLUDE = r"REAL ESTATE|REALTY|ROYALTY|MOTOR CREDIT|NORTHERN TRUST|ACQUISITION|TAX CREDIT"


def fund_like(names):
    up = pd.Series(names).str.upper()
    return (up.str.contains(FUND_PATTERN, regex=True, na=False) & ~up.str.contains(FUND_EXCLUDE, regex=True, na=False)).values


def forecast_candidates(D, usable, s=Settings(), window_days=200):
    """Each company's latest usable filing in the analysis window: the rows whose next quarter is not known yet."""
    latest = D.loc[D.groupby("cik")["period"].idxmax()]
    last_q_start = pd.Period(s.analysis_last_quarter.upper(), freq="Q").start_time
    keep = usable.loc[latest.index] & (latest["period"] >= last_q_start - pd.Timedelta(days=window_days))
    return latest[keep].copy()


def final_test_rows(cand, new_raw, s=Settings()):
    """Attach the outcome from NEWER quarter files to the registered forecasts, with the same rules as the label (75 to 110 days,
    reported three-month figure or cumulative minus the previous cumulative of exactly one quarter fewer)."""
    new_raw = new_raw.sort_values(["cik", "period", "days_to_file"]).drop_duplicates(["cik", "period"], keep="last")
    nx = new_raw[["cik", "period", "dur_qtrs", "days_to_file", "NetIncomeLoss", "NetIncomeLoss_qtr", "NetIncomeLoss_qtr_src"]]
    m = (cand[["cik", "period", "dur_qtrs", "NetIncomeLoss"]].reset_index()
         .merge(nx.rename(columns={c: "n_" + c for c in nx.columns if c != "cik"}), on="cik"))
    m["gap"] = (m["n_period"] - m["period"]).dt.days
    m = m[m["gap"].between(*s.gap_range)].sort_values("gap").drop_duplicates("index")
    chain = np.where(m["n_dur_qtrs"] == 1, m["n_NetIncomeLoss"],
                     np.where(m["n_dur_qtrs"] == m["dur_qtrs"] + 1, m["n_NetIncomeLoss"] - m["NetIncomeLoss"], np.nan))
    m["ni_next"] = np.where(m["n_NetIncomeLoss_qtr_src"].eq("reported"), m["n_NetIncomeLoss_qtr"], chain)
    m = m[pd.notna(m["ni_next"]) & m["n_days_to_file"].between(0, s.max_days_to_file)]
    out = cand.loc[m["index"].values].copy()
    out["loss_next"] = (m["ni_next"].values < 0).astype(float)
    return out


def screening_list(forecasts, threshold):
    """One row per company, riskiest first, with the rank among profitable-now companies (the informative transition list)."""
    out = forecasts.sort_values("p_loss_next", ascending=False).reset_index(drop=True)
    out["flag"] = out["p_loss_next"] >= threshold
    prof = out["loss_now"] == 0
    out["rank_profitable_now"] = np.where(prof, prof.cumsum(), np.nan)
    return out

"""The standalone quarterly profit and the next-quarter label (notebook Cells 3 to 5).

The key rule: a quarter's profit is rebuilt from the CUMULATIVE year-to-date figure by period order. The previous row must end
75 to 110 days earlier and cover exactly one quarter fewer; chaining by fiscal-year label can subtract the wrong fiscal year."""
import numpy as np
import pandas as pd

from .config import Settings


def one_filing_per_quarter(raw):
    """Keep the latest filed version of each (cik, period) and sort by company and period."""
    out = raw.sort_values(["cik", "period", "days_to_file"]).drop_duplicates(["cik", "period"], keep="last")
    return out.sort_values(["cik", "period"]).reset_index(drop=True)


def quarter_links(d, s=Settings()):
    """Boolean Series: does each row have a true previous / next quarter (75 to 110 days away)?"""
    g = d.groupby("cik")
    gap_prev = (d["period"] - g["period"].shift(1)).dt.days
    gap_next = (g["period"].shift(-1) - d["period"]).dt.days
    return gap_prev.between(*s.gap_range), gap_next.between(*s.gap_range)


def standalone_quarter(d, prev_ok):
    """The three-month net income: the reported figure if the filing has one, otherwise the period-chain difference."""
    g = d.groupby("cik")
    prev_cum, prev_dur = g["NetIncomeLoss"].shift(1), g["dur_qtrs"].shift(1)
    chain_ok = prev_ok & (prev_dur == d["dur_qtrs"] - 1)
    chain = np.where(d["dur_qtrs"] == 1, d["NetIncomeLoss"], np.where(chain_ok, d["NetIncomeLoss"] - prev_cum, np.nan))
    return pd.Series(np.where(d["NetIncomeLoss_qtr_src"].eq("reported"), d["NetIncomeLoss_qtr"], chain), index=d.index, name="ni_q")


def add_labels(d, next_ok):
    """Next quarter's profit and the S1 label (1 = net loss next quarter); blank without a true next quarter."""
    g = d.groupby("cik")
    out = d.copy()
    out["ni_q_next"] = g["ni_q"].shift(-1).where(next_ok)
    out["next_period"] = g["period"].shift(-1).where(next_ok)
    out["loss_now"] = (out["ni_q"] < 0).astype(float).where(out["ni_q"].notna())
    out["loss_next"] = (out["ni_q_next"] < 0).astype(float).where(out["ni_q_next"].notna())
    out["filed_date"] = out["period"] + pd.to_timedelta(out["days_to_file"], unit="D")
    out["label_known_date"] = out["next_period"] + pd.to_timedelta(g["days_to_file"].shift(-1).where(next_ok), unit="D")
    return out


def build_labelled_panel(raw, s=Settings()):
    """raw filings -> one row per company-quarter with ni_q, the label and the link flags."""
    d = one_filing_per_quarter(raw)
    prev_ok, next_ok = quarter_links(d, s)
    d["ni_q"] = standalone_quarter(d, prev_ok)
    d = add_labels(d, next_ok)
    d["prev_ok"], d["next_ok"] = prev_ok, next_ok
    return d

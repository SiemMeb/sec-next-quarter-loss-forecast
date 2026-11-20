"""The 30 S1 features (notebook Cell 5). Every feature uses only the current filing and earlier, chained quarters."""
import numpy as np
import pandas as pd

from .config import Settings

CAT_COLS = ["next_fq", "div_code"]
FLAG_COLS = ["loss_now", "loss_lag1", "loss_lag2", "loss_lag3", "loss_streak", "is_spac", "is_shell"]


def add_lags(d, prev_ok):
    """Profit and revenue of the previous 1 to 4 quarters, only through an unbroken chain of true consecutive quarters."""
    out = d.copy()
    g = out.groupby("cik")
    ok_g = prev_ok.groupby(out["cik"])
    valid = {1: prev_ok}
    for k in (2, 3, 4):
        valid[k] = valid[k - 1] & ok_g.shift(k - 1, fill_value=False)
    for k in range(1, 5):
        out[f"lag_ni_{k}"] = g["ni_q"].shift(k).where(valid[k])
        out[f"lag_rev_{k}"] = g["Revenues_qtr"].shift(k).where(valid[k])
    return out


def build_features(d, s=Settings()):
    """Feature frame F (same index as d). `d` must already hold ni_q, loss_now and the lag columns."""
    A = d["Assets"].where(d["Assets"] >= s.min_assets)
    ratio = lambda num, den: num / den.where(den > 0)
    non_neg = lambda x: x.where(x >= 0)
    signed_log = lambda x: np.sign(x) * np.log10(1 + np.abs(x))

    F = pd.DataFrame(index=d.index)
    F["ni_a0"] = d["ni_q"] / A
    for k in range(1, 5):
        F[f"ni_a{k}"] = d[f"lag_ni_{k}"] / A
    F["loss_now"] = d["loss_now"]
    for k in (1, 2, 3):
        F[f"loss_lag{k}"] = (d[f"lag_ni_{k}"] < 0).astype(float).where(d[f"lag_ni_{k}"].notna())
    alive = F["loss_now"] == 1
    streak = alive.astype(int)
    for k in (1, 2, 3):
        alive = alive & (F[f"loss_lag{k}"] == 1)
        streak = streak + alive.astype(int)
    F["loss_streak"] = streak.where(F["loss_now"].notna())
    F["roa_ttm"] = d["NetIncomeLoss_ttm"] / A
    F["net_margin_q"] = ratio(d["ni_q"], d["Revenues_qtr"])
    F["op_margin_q"] = ratio(d["OperatingIncomeLoss_qtr"], d["Revenues_qtr"])
    F["rev_qoq"] = d["Revenues_qtr"] / d["lag_rev_1"].where(d["lag_rev_1"] > 0) - 1
    F["rev_yoy"] = d["Revenues_qtr"] / d["lag_rev_4"].where(d["lag_rev_4"] > 0) - 1
    F["ocf_a"] = d["NetCashProvidedByUsedInOperatingActivities_qtr"] / A
    F["accrual_a"] = (d["ni_q"] - d["NetCashProvidedByUsedInOperatingActivities_qtr"]) / A
    F["financing_a"] = d["NetCashProvidedByUsedInFinancingActivities_qtr"] / A
    F["debt_to_assets"] = ratio(non_neg(d["Liabilities"]), A)
    F["current_ratio"] = ratio(non_neg(d["AssetsCurrent"]), d["LiabilitiesCurrent"])
    F["cash_to_cur_liab"] = ratio(non_neg(d["CashAndCashEquivalentsAtCarryingValue"]), d["LiabilitiesCurrent"])
    F["equity_ratio"] = d["EquityTotal"] / A
    F["retained_a"] = d["RetainedEarningsAccumulatedDeficit"] / A
    F["log_assets"] = np.log10(d["Assets"].where(d["Assets"] > 0))
    F["log_rev_q"] = signed_log(d["Revenues_qtr"])
    F["days_to_file"] = d["days_to_file"].clip(0, s.max_days_to_file)
    F["is_spac"] = d["is_spac"]
    F["is_shell"] = d["is_shell"]          # no amendment flag: whether a filing gets amended is only known later
    F["next_fq"] = d["fp"].map({"Q1": 2, "Q2": 3, "Q3": 4, "FY": 1}).astype(float)
    F["div_code"] = pd.Series(pd.Categorical(d["sic_division"]).codes, index=d.index).astype(float).where(lambda x: x >= 0)
    return F


def feature_lists(F):
    """(FEATURES, CLIP_COLS): all feature names, and the continuous ones that are clipped."""
    features = list(F.columns)
    return features, [c for c in features if c not in CAT_COLS + FLAG_COLS]


def model_frame(d, F, s=Settings()):
    """The modelling panel D (d plus f_-prefixed features) and the row filter of notebook Cell 6, without the label condition."""
    D = pd.concat([d, F.add_prefix("f_")], axis=1)
    usable = (D["loss_now"].notna() & (D["period"] >= pd.Timestamp(s.window_start))
              & D["days_to_file"].between(0, s.max_days_to_file) & (D["has_negative_nonneg_tag"].fillna(0) == 0))
    return D, usable


def time_split(D, usable, s=Settings()):
    """Labelled usable rows with a `split` column: train / valid / test by the quarter of the LABEL."""
    S = D[usable & D["loss_next"].notna()].copy()
    S["split"] = np.where(S["next_period"] <= pd.Timestamp(s.train_end), "train",
                          np.where(S["next_period"] <= pd.Timestamp(s.valid_end), "valid", "test"))
    return S

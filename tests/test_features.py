import numpy as np
import pandas as pd

from lossforecast.features import add_lags, build_features, feature_lists, model_frame, time_split
from lossforecast.labels import build_labelled_panel


def features_for(raw):
    d = build_labelled_panel(raw)
    d = add_lags(d, d["prev_ok"])
    return d, build_features(d)


def test_feature_set(panel):
    d, F = features_for(panel)
    features, clip_cols = feature_lists(F)
    assert len(features) == 30 and "is_amended" not in features and not any("EarningsPerShare" in f for f in features)
    assert set(clip_cols).isdisjoint({"next_fq", "div_code", "loss_now", "is_spac"})


def test_no_look_ahead(panel):
    """Features of a filing must not change when every LATER filing is removed."""
    cut = pd.Timestamp("2025-06-30")
    d_all, F_all = features_for(panel)
    d_cut, F_cut = features_for(panel[panel["period"] <= cut])
    key = ["cik", "period"]
    a = pd.concat([d_all[key], F_all], axis=1).set_index(key)
    b = pd.concat([d_cut[key], F_cut], axis=1).set_index(key)
    a = a.loc[b.index]
    pd.testing.assert_frame_equal(a.drop(columns="div_code"), b.drop(columns="div_code"), check_exact=False, rtol=1e-12)


def test_time_split_order(panel):
    d, F = features_for(panel)
    D, usable = model_frame(d, F)
    S = time_split(D, usable)
    tr, va, te = (S[S["split"] == k] for k in ("train", "valid", "test"))
    assert tr["next_period"].max() < va["next_period"].min() <= va["next_period"].max() < te["next_period"].min()
    assert S["days_to_file"].between(0, 365).all() and S["loss_next"].notna().all()


def test_loss_streak_counts_consecutive_losses():
    rows = [dict(cik=1, period=p, dur_qtrs=1.0, NetIncomeLoss=v, NetIncomeLoss_qtr=v, NetIncomeLoss_qtr_src="reported", days_to_file=30.0)
            for p, v in zip(pd.date_range("2024-03-31", periods=4, freq="QE"), [5.0, -1.0, -2.0, -3.0])]
    raw = pd.DataFrame(rows)
    for c in ["Assets", "Revenues_qtr", "NetIncomeLoss_ttm", "OperatingIncomeLoss_qtr", "NetCashProvidedByUsedInOperatingActivities_qtr",
              "NetCashProvidedByUsedInFinancingActivities_qtr", "Liabilities", "AssetsCurrent", "LiabilitiesCurrent",
              "CashAndCashEquivalentsAtCarryingValue", "EquityTotal", "RetainedEarningsAccumulatedDeficit", "is_spac", "is_shell"]:
        raw[c] = 1e7 if c == "Assets" else np.nan
    raw["fp"], raw["sic_division"] = "Q1", "I: Services"
    d, F = features_for(raw)
    assert F["loss_streak"].tolist() == [0, 1, 2, 3]

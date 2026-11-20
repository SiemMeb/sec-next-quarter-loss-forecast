import numpy as np
import pandas as pd

from lossforecast.labels import build_labelled_panel, one_filing_per_quarter


def frame(rows):
    base = dict(cik=1, NetIncomeLoss_qtr=np.nan, NetIncomeLoss_qtr_src="derived", days_to_file=40.0)
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["period"] = pd.to_datetime(df["period"])
    return df


def test_standalone_quarter_by_period_chain():
    # fiscal year = calendar year; cumulative 10, 25, 45, annual 50 -> quarters 10, 15, 20, 5
    d = build_labelled_panel(frame([dict(period="2024-03-31", dur_qtrs=1, NetIncomeLoss=10.0),
                                    dict(period="2024-06-30", dur_qtrs=2, NetIncomeLoss=25.0),
                                    dict(period="2024-09-30", dur_qtrs=3, NetIncomeLoss=45.0),
                                    dict(period="2024-12-31", dur_qtrs=4, NetIncomeLoss=50.0)]))
    assert d["ni_q"].tolist() == [10.0, 15.0, 20.0, 5.0]


def test_reported_three_month_figure_wins():
    d = build_labelled_panel(frame([dict(period="2024-03-31", dur_qtrs=1, NetIncomeLoss=10.0),
                                    dict(period="2024-06-30", dur_qtrs=2, NetIncomeLoss=25.0, NetIncomeLoss_qtr=-3.0,
                                         NetIncomeLoss_qtr_src="reported")]))
    assert d["ni_q"].iloc[1] == -3.0


def test_no_chain_across_a_missing_quarter():
    # Q2 is missing: Q3 (nine months) cannot be differenced against Q1 (three months)
    d = build_labelled_panel(frame([dict(period="2024-03-31", dur_qtrs=1, NetIncomeLoss=10.0),
                                    dict(period="2024-09-30", dur_qtrs=3, NetIncomeLoss=45.0)]))
    assert np.isnan(d["ni_q"].iloc[1])
    assert np.isnan(d["loss_next"].iloc[0])          # no true next quarter either (183 days)


def test_no_chain_across_fiscal_years():
    # a 10-K (12 months) followed by a nine-month 10-Q: durations do not step by one, so no difference is taken
    d = build_labelled_panel(frame([dict(period="2024-12-31", dur_qtrs=4, NetIncomeLoss=72.9),
                                    dict(period="2025-03-31", dur_qtrs=3, NetIncomeLoss=17.5)]))
    assert np.isnan(d["ni_q"].iloc[1])


def test_label_is_next_quarter_loss():
    d = build_labelled_panel(frame([dict(period="2024-03-31", dur_qtrs=1, NetIncomeLoss=10.0),
                                    dict(period="2024-06-30", dur_qtrs=2, NetIncomeLoss=5.0)]))
    assert d["loss_now"].tolist() == [0.0, 1.0]
    assert d["loss_next"].iloc[0] == 1.0 and np.isnan(d["loss_next"].iloc[1])
    assert d["label_known_date"].iloc[0] == pd.Timestamp("2024-06-30") + pd.Timedelta(days=40)


def test_latest_filed_duplicate_is_kept():
    raw = frame([dict(period="2024-03-31", dur_qtrs=1, NetIncomeLoss=10.0, days_to_file=40.0),
                 dict(period="2024-03-31", dur_qtrs=1, NetIncomeLoss=12.0, days_to_file=95.0)])
    out = one_filing_per_quarter(raw)
    assert len(out) == 1 and out["NetIncomeLoss"].iloc[0] == 12.0


def test_panel_invariants(panel):
    d = build_labelled_panel(panel)
    assert not d.duplicated(["cik", "period"]).any()
    gaps = (d["next_period"] - d["period"]).dt.days.dropna()
    assert gaps.between(75, 110).all()
    assert (d["loss_next"].notna() <= d["next_ok"]).all()

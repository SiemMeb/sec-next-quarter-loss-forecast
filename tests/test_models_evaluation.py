import numpy as np
import pandas as pd

from lossforecast.evaluation import cluster_bootstrap, cost_per_1000, flag_stats, thr_best_f1, thr_min_cost, thr_min_precision, thr_min_recall
from lossforecast.models import Clipper, fit_predict


def test_clip_limits_come_from_training_rows_only():
    tr = pd.DataFrame({"f_x": np.arange(100, dtype=float)})
    te = pd.DataFrame({"f_x": [-1e9, 50.0, 1e9]})
    c = Clipper(["x"], ["x"]).fit(tr)
    out = c.transform(te)["x"].values
    assert out[0] == tr["f_x"].quantile(0.01) and out[2] == tr["f_x"].quantile(0.99) and out[1] == 50.0
    c2 = Clipper(["x"], ["x"]).fit(tr)
    c2.transform(pd.concat([te] * 10))                 # transforming other data never moves the limits
    assert c2.lo.equals(c.lo) and c2.hi.equals(c.hi)


def test_fit_predict_is_deterministic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f_a": rng.normal(size=600), "f_b": rng.normal(size=600)})
    df["y"] = (df["f_a"] + rng.normal(scale=0.5, size=600) > 0).astype(float)
    p1, _, _ = fit_predict(df.iloc[:400], [df.iloc[400:]], ["a", "b"], ["a", "b"], target="y")
    p2, _, _ = fit_predict(df.iloc[:400], [df.iloc[400:]], ["a", "b"], ["a", "b"], target="y")
    assert np.array_equal(p1[0], p2[0])


def test_cluster_bootstrap_keeps_companies_together():
    cik = np.repeat(np.arange(50), 4)
    seen = []
    point, draws = cluster_bootstrap(cik, {"n": lambda i: (seen.append(i), len(i))[1]}, n_boot=20)
    assert point["n"] == 200 and len(draws["n"]) == 20
    for idx in seen[1:]:
        counts = pd.Series(cik[idx]).value_counts()
        assert (counts % 4 == 0).all()                 # every drawn company brings all four of its filings


def test_thresholds():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1]); s = np.linspace(0.1, 0.8, 8)
    assert flag_stats(y, s >= thr_min_precision(y, s, 0.75))["precision"] >= 0.75
    assert flag_stats(y, s >= thr_min_recall(y, s, 1.0))["recall"] == 1.0
    assert 0.1 <= thr_best_f1(y, s) <= 0.8


def test_costs():
    y = np.array([1, 0, 0, 0])
    assert cost_per_1000(y, np.zeros(4, bool), 5) == 1250          # one miss at cost 5, per 1,000 filings
    assert cost_per_1000(y, np.ones(4, bool), 5) == 750            # three false alarms
    s = np.array([0.9, 0.1, 0.2, 0.3])
    assert flag_stats(y, s >= thr_min_cost(y, s, 5))["fp"] == 0    # a perfect ranking: flag exactly the event

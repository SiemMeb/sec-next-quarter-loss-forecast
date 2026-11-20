"""Scores, company-clustered bootstrap intervals, thresholds and error costs (notebook Cells 7, 19, 20 and 20b)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    return dict(n=len(y), loss_rate=y.mean(), roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def cluster_bootstrap(cik, stats, n_boot=500, seed=42):
    """Resample COMPANIES with replacement (all filings of a drawn company come along) and recompute each statistic.
    `stats` maps a name to a function of a row-position array. Returns (point estimates, draws)."""
    codes, uniq = pd.factorize(pd.Series(cik))
    order = np.argsort(codes, kind="mergesort")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = {k: f(np.arange(len(codes))) for k, f in stats.items()}
    draws = {k: [] for k in stats}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(uniq), len(uniq))])
        for k, f in stats.items():
            try:
                draws[k].append(f(idx))
            except ValueError:                       # a resample with one class only
                draws[k].append(np.nan)
    return point, {k: np.asarray(v, dtype=float) for k, v in draws.items()}


def interval(draws, level=0.95):
    a = (1 - level) / 2 * 100
    return tuple(np.nanpercentile(draws, [a, 100 - a]))


def flag_stats(y, flag):
    y = np.asarray(y).astype(int); flag = np.asarray(flag, dtype=bool)
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum()); fn = int((~flag & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(flag_rate=flag.mean(), precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12),
                tp=tp, fp=fp, fn=fn, tn=len(y) - tp - fp - fn)


def thr_best_f1(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f)])


def thr_min_precision(y, s, target):
    """The lowest threshold that still reaches the target precision (the most recall at that precision)."""
    p, r, t = precision_recall_curve(y, s)
    ok = np.where(p[:-1] >= target)[0]
    return float(t[ok[0]]) if len(ok) else float("inf")


def thr_min_recall(y, s, target):
    """The highest threshold that still reaches the target recall (the fewest flags for that recall)."""
    p, r, t = precision_recall_curve(y, s)
    ok = np.where(r[:-1] >= target)[0]
    return float(t[ok[-1]]) if len(ok) else float("-inf")


def cost_per_1000(y, flag, cost_ratio):
    """Expected cost per 1,000 filings in units of one false alarm, when a missed event costs `cost_ratio` false alarms."""
    y = np.asarray(y).astype(int); flag = np.asarray(flag, dtype=bool)
    return 1000 * (cost_ratio * ((~flag) & (y == 1)).sum() + (flag & (y == 0)).sum()) / len(y)


def thr_min_cost(y, s, cost_ratio, grid=401):
    """The threshold with the lowest expected cost on (y, s); +inf (flag nothing) is a candidate."""
    cand = np.unique(np.r_[np.quantile(s, np.linspace(0, 1, grid)), np.inf])
    return float(cand[int(np.argmin([cost_per_1000(y, s >= t, cost_ratio) for t in cand]))])

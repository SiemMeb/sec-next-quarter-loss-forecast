"""End-to-end S1 run: python -m lossforecast.pipeline --data <folder with the quarter files> [--out results]

Reproduces the headline comparison of the notebook (persistence, logistic regression, gradient boosting on the label-quarter
split) and writes the registered forecasts for the companies whose next quarter is not yet in the data."""
import argparse
import os

import pandas as pd

from .config import Settings
from .data import load_quarters, quarter_files
from .evaluation import metrics, thr_best_f1
from .features import add_lags, build_features, feature_lists, model_frame, time_split
from .labels import build_labelled_panel
from .models import Clipper, make_gbm, make_logit
from .screening import forecast_candidates, fund_like, screening_list


def run(raw, s=Settings()):
    d = build_labelled_panel(raw, s)
    d = add_lags(d, d["prev_ok"])
    F = build_features(d, s)
    features, clip_cols = feature_lists(F)
    D, usable = model_frame(d, F, s)
    S = time_split(D, usable, s)
    tr, va, te = (S[S["split"] == k] for k in ("train", "valid", "test"))
    clipper = Clipper(features, clip_cols).fit(tr)
    Xtr, Xva, Xte = (clipper.transform(x) for x in (tr, va, te))
    gbm = make_gbm(features, seed=s.seed).fit(Xtr, tr["loss_next"].values)
    logit = make_logit(features).fit(Xtr, tr["loss_next"].values)
    rows = []
    for split, df, X in (("valid", va, Xva), ("test", te, Xte)):
        y = df["loss_next"].values
        rows.append(dict(split=split, model="persistence", **metrics(y, df["f_loss_now"].values)))
        rows.append(dict(split=split, model="logistic", **metrics(y, logit.predict_proba(X)[:, 1])))
        rows.append(dict(split=split, model="gradient boosting", **metrics(y, gbm.predict_proba(X)[:, 1])))
    results = pd.DataFrame(rows)
    threshold = thr_best_f1(va["loss_next"].values, gbm.predict_proba(Xva)[:, 1])
    cand = forecast_candidates(D, usable, s)
    cand["p_loss_next"] = gbm.predict_proba(clipper.transform(cand))[:, 1]
    cand["loss_now"] = cand["f_loss_now"]
    cand["fund_like"] = fund_like(cand["name"])
    screen = screening_list(cand[["adsh", "cik", "name", "form", "fp", "period", "filed_date", "loss_now", "p_loss_next", "fund_like", "is_spac"]],
                            threshold)
    return dict(results=results, threshold=threshold, screen=screen, model=gbm, clipper=clipper, features=features)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="folder with 2023q1.parquet ... 2026q2.parquet")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    s = Settings()
    files, _ = quarter_files(a.data, s.analysis_last_quarter)
    out = run(load_quarters(files), s)
    os.makedirs(a.out, exist_ok=True)
    out["results"].round(4).to_csv(os.path.join(a.out, "s1_results.csv"), index=False)
    out["screen"].to_csv(os.path.join(a.out, "screening_list.csv"), index=False)
    print(out["results"].round(4).to_string(index=False))
    print(f"threshold (best F1 on validation): {out['threshold']:.3f}; screening list: {len(out['screen']):,} companies -> {a.out}/")


if __name__ == "__main__":
    main()

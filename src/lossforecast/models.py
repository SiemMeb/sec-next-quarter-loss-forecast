"""Models and the leakage-safe preparation (notebook Cells 8 and 12b): clip limits are always learned from training rows only."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CAT_COLS


@dataclass
class Clipper:
    """1st/99th percentile caps fitted on a training frame, applied unchanged to any other frame."""
    features: list
    clip_cols: list
    lo: pd.Series = None
    hi: pd.Series = None

    def fit(self, train_df):
        cc = ["f_" + c for c in self.clip_cols if c in self.features]
        self.lo, self.hi = train_df[cc].quantile(0.01), train_df[cc].quantile(0.99)
        return self

    def transform(self, df):
        X = df[["f_" + c for c in self.features]].copy()
        X[self.lo.index] = X[self.lo.index].clip(lower=self.lo, upper=self.hi, axis=1)
        X.columns = self.features
        return X


def make_gbm(features, cat_cols=CAT_COLS, seed=42, learning_rate=0.06, max_iter=400, n_iter_no_change=20):
    """The headline gradient-boosting model (Cell 8 settings; Cell 12b's fit_predict uses learning_rate 0.08, max_iter 250, 15)."""
    cats = [features.index(c) for c in cat_cols if c in features] or None
    return HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, categorical_features=cats, early_stopping=True,
                                          n_iter_no_change=n_iter_no_change, validation_fraction=0.1, random_state=seed)


def make_logit(features, cat_cols=CAT_COLS, C=0.5, max_iter=1000):
    """Median imputation with missing flags, scaling, one-hot categoricals, L2 logistic regression."""
    num = [c for c in features if c not in cat_cols]
    cat = [c for c in features if c in cat_cols]
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return Pipeline([("prep", ColumnTransformer(parts)), ("clf", LogisticRegression(C=C, max_iter=max_iter))])


def fit_predict(train_df, eval_dfs, features, clip_cols, target="loss_next", seed=42, **gbm_kwargs):
    """Fit gradient boosting on train_df (clip limits from train_df only) and return (predictions per eval frame, model, clipper)."""
    clipper = Clipper(list(features), list(clip_cols)).fit(train_df)
    model = make_gbm(list(features), seed=seed, **gbm_kwargs).fit(clipper.transform(train_df), train_df[target].values)
    return [model.predict_proba(clipper.transform(df))[:, 1] for df in eval_dfs], model, clipper

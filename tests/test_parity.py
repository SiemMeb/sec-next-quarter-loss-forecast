"""The modules must reproduce the notebook: run the notebook's own cells 1 to 8 on the synthetic panel and compare."""
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from lossforecast.features import add_lags, build_features, feature_lists, model_frame, time_split
from lossforecast.labels import build_labelled_panel
from lossforecast.models import Clipper, make_gbm

NB = os.path.join(os.path.dirname(__file__), "..", "notebooks", "next-quarter-loss-forecast.ipynb")


@pytest.fixture(scope="module")
def notebook_state(panel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: plt.close("all")
    stubbed = []
    for m in ("pyarrow", "pyarrow.parquet", "pyarrow.dataset"):        # only if pyarrow is not installed: Cell 1 imports it
        if m not in sys.modules:
            try:
                __import__(m)
            except ImportError:
                sys.modules[m] = types.ModuleType(m); stubbed.append(m)
    cells = [("".join(c["source"])) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
    ns = {"_RAW": panel}
    for s in cells:
        if s.startswith("parquet_files = sorted("):
            s = "raw = _RAW.copy()\nnewer_files = []\n"
        exec(compile(s, "notebook", "exec"), ns)
        for m in stubbed:                               # drop the stub once Cell 1 has run, so other libraries never see it
            sys.modules.pop(m, None)
        stubbed = []
        if "gbm.fit(Xtr, ytr)" in s:                   # Cell 8: models fitted, stop here
            break
    return ns


def test_parity_with_notebook(panel, notebook_state):
    nb = notebook_state
    d = build_labelled_panel(panel)
    d = add_lags(d, d["prev_ok"])
    F = build_features(d)
    features, clip_cols = feature_lists(F)
    assert features == nb["FEATURES"] and clip_cols == nb["CLIP_COLS"]
    pd.testing.assert_series_equal(d["ni_q"], nb["d"]["ni_q"], check_names=False)
    pd.testing.assert_series_equal(d["loss_next"], nb["d"]["loss_next"], check_names=False)
    pd.testing.assert_frame_equal(F, nb["F"])
    D, usable = model_frame(d, F)
    S = time_split(D, usable)
    assert S.index.equals(nb["S"].index) and (S["split"] == nb["S"]["split"]).all()
    tr, te = S[S["split"] == "train"], S[S["split"] == "test"]
    clipper = Clipper(features, clip_cols).fit(tr)
    gbm = make_gbm(features).fit(clipper.transform(tr), tr["loss_next"].values)
    np.testing.assert_allclose(gbm.predict_proba(clipper.transform(te))[:, 1], nb["pred"][("gradient boosting", "test")])

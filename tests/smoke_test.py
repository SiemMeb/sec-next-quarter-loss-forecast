"""Smoke test: run every code cell of the notebook on a synthetic panel with the dataset's schema.
pyarrow is not installed here, so Cell 2's loading is replaced by the synthetic frame and parquet writes go to pickle."""
import json, sys, types, os, time, traceback
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")

HERE = os.path.dirname(os.path.abspath(__file__))
NB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "notebooks", "next-quarter-loss-forecast.ipynb")
WORK = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "_work")
FINAL = os.environ.get("FINAL_TEST") == "1"     # also run the final-test branch of Cell 21b
os.makedirs(WORK, exist_ok=True)

# ---- stub pyarrow so Cell 1's imports succeed
for m in ("pyarrow", "pyarrow.parquet", "pyarrow.dataset"):
    sys.modules[m] = types.ModuleType(m)
sys.modules["pyarrow"].parquet = sys.modules["pyarrow.parquet"]; sys.modules["pyarrow"].dataset = sys.modules["pyarrow.dataset"]
pd.DataFrame.to_parquet = lambda self, path, index=False, **k: self.to_pickle(path + ".pkl")

# ---- stub IPython.display (Kaggle has it; this container does not)
_ip = types.ModuleType("IPython"); _ipd = types.ModuleType("IPython.display")
class Markdown(str): pass
_ipd.Markdown = Markdown; _ipd.display = lambda x: print("[DISPLAY MARKDOWN]\n" + str(x))
_ip.display = _ipd; _ip.get_ipython = lambda: None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import make_panel
SYN = make_panel()
print("synthetic rows:", SYN.shape)

# ---- run the cells
nb = json.load(open(NB))
ns = {}
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
for i, s in code_cells:
    s = s.replace("/kaggle/working", WORK)
    if s.startswith("parquet_files = sorted("):          # Cell 2: use the synthetic frame instead of reading parquet
        # FINAL_TEST=1 holds the last synthetic quarter back as a "newer quarter file", to exercise the final test in Cell 21b
        s = ("def load_quarters(frames):\n    out = pd.concat(frames, ignore_index=True); out['period'] = pd.to_datetime(out['period'])\n"
             "    for c in ('is_spac', 'is_shell', 'is_amended', 'has_negative_nonneg_tag'):\n        out[c] = out[c].astype('boolean').astype('float')\n"
             "    return out\n"
             "raw = load_quarters([_SYN[_SYN['period'] < _CUT]])\n"
             "newer_files = [_SYN[_SYN['period'] >= _CUT]] if _FINAL else []\n"
             "print('loaded:', raw.shape, '| quarters held back for the final test:', len(newer_files))\n")
        ns["_SYN"] = SYN; ns["_FINAL"] = FINAL
        ns["_CUT"] = SYN["period"].max() if FINAL else SYN["period"].max() + pd.Timedelta(days=1)
    if "from IPython.display import" in s:
        sys.modules["IPython"] = _ip; sys.modules["IPython.display"] = _ipd
    t0 = time.time()
    print(f"\n######## cell {i} ########")
    try:
        exec(compile(s, f"cell{i}", "exec"), ns)
    except Exception:
        traceback.print_exc()
        print(f"FAILED at cell {i}")
        sys.exit(1)
    print(f"[cell {i}: {time.time() - t0:.1f}s]")
    if i == 2:                                       # imports done: drop the stub so other libraries do not see a fake pyarrow
        for m in ("pyarrow", "pyarrow.parquet", "pyarrow.dataset"):
            sys.modules.pop(m, None)
print("\nALL CELLS RAN")

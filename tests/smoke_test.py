"""Run every code cell of the notebook on a synthetic panel with the dataset's schema.
pyarrow is not installed here, so Cell 2's loading is replaced by the synthetic frame and parquet writes go to pickle."""
import json, sys, types, os, time, traceback
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")

HERE = os.path.dirname(os.path.abspath(__file__))
NB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "notebooks", "next-quarter-loss-forecast.ipynb")
WORK = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "_work")
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

# ---- synthetic panel
rng = np.random.default_rng(0)
N = 2500
periods = pd.date_range("2022-09-30", "2026-03-31", freq="QE")
rows = []
divs = ["B: Mining", "D: Manufacturing", "E: Transportation, communications, utilities", "G: Retail trade", "H: Finance, insurance, real estate", "I: Services"]
for i in range(N):
    kind = rng.choice(["op", "spac", "fund", "shell"], p=[0.86, 0.06, 0.04, 0.04])
    name = {"op": f"COMPANY {i} INC", "spac": f"BLUE {i} ACQUISITION CORP", "fund": f"ALPHA {i} BDC, INC.", "shell": f"SHELL {i} HOLDINGS"}[kind]
    assets = float(np.exp(rng.normal(19, 2.2)))
    mu = rng.normal(0.004, 0.02); x = mu + rng.normal(0, 0.02)
    eq_ratio = float(np.clip(rng.normal(0.35, 0.3), -0.3, 0.95))
    start = rng.integers(0, 4); stop = len(periods) - rng.integers(0, 3)
    cum = 0.0; ttm = []
    for t in range(start, stop):
        if rng.random() < 0.03:      # a missing filing (gap)
            continue
        p = periods[t]; q = p.quarter
        shock = rng.normal(0, 0.015) + (0.03 if (kind == "fund" and p >= pd.Timestamp("2026-01-01")) else 0)
        x = 0.7 * x + 0.3 * mu + shock - (0.02 if kind == "fund" and p >= pd.Timestamp("2026-01-01") else 0)
        ni = assets * x * (1 + 0.3 * (q == 4))
        assets *= float(np.exp(rng.normal(0.01, 0.03)))
        eq_ratio = eq_ratio + x * 0.8 + rng.normal(0, 0.01)
        equity = assets * eq_ratio
        cum = ni if q == 1 else cum + ni
        ttm = (ttm + [ni])[-4:]
        rev = assets * abs(rng.normal(0.08, 0.03)) if kind == "op" and rng.random() > 0.1 else np.nan
        liab = assets - equity
        rows.append(dict(adsh=f"{i:06d}-{t:02d}", cik=i, name=name, sic_division=rng.choice(divs) if kind == "op" else divs[4],
                         form="10-K" if q == 4 else "10-Q", fp="FY" if q == 4 else f"Q{q}", period=p, dur_qtrs=float(q),
                         days_to_file=float(rng.integers(20, 90)) if rng.random() > 0.01 else float(rng.choice([-30, 500])),
                         is_spac=kind == "spac", is_shell=kind == "shell", is_amended=rng.random() < 0.02,
                         has_negative_nonneg_tag=rng.random() < 0.001,
                         NetIncomeLoss_qtr_src="derived" if q == 4 else "reported",
                         Assets=assets, AssetsCurrent=assets * 0.4, Liabilities=liab, LiabilitiesCurrent=max(liab, 1) * 0.5,
                         EquityTotal=equity, RetainedEarningsAccumulatedDeficit=equity - assets * 0.2,
                         CashAndCashEquivalentsAtCarryingValue=assets * 0.1,
                         NetIncomeLoss=cum, NetIncomeLoss_qtr=ni, NetIncomeLoss_ttm=sum(ttm) if len(ttm) == 4 else np.nan,
                         Revenues_qtr=rev, OperatingIncomeLoss_qtr=ni * 1.2 if rev == rev else np.nan,
                         NetCashProvidedByUsedInOperatingActivities_qtr=ni + assets * rng.normal(0.005, 0.01),
                         NetCashProvidedByUsedInFinancingActivities_qtr=assets * rng.normal(0, 0.01)))
SYN = pd.DataFrame(rows)
SYN = pd.concat([SYN, SYN.sample(9, random_state=1).assign(days_to_file=lambda d: d["days_to_file"] + 5)], ignore_index=True)  # 9 duplicates
print("synthetic rows:", SYN.shape)

# ---- run the cells
nb = json.load(open(NB))
ns = {}
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
for i, s in code_cells:
    s = s.replace("/kaggle/working", WORK)
    if s.startswith("parquet_files = sorted("):          # Cell 2: use the synthetic frame instead of reading parquet
        s = ("raw = _SYN.copy()\nraw['period'] = pd.to_datetime(raw['period'])\n"
             "for c in ('is_spac', 'is_shell', 'is_amended', 'has_negative_nonneg_tag'):\n    raw[c] = raw[c].astype('boolean').astype('float')\n"
             "print('loaded:', raw.shape)\n")
        ns["_SYN"] = SYN
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

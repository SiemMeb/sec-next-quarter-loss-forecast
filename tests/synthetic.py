"""A synthetic filing panel with the dataset's schema: about 2,500 companies (operating firms, SPACs, BDC-like funds, shells),
quarter-ends 2022-09-30 to 2026-03-31, missing filings, late and negative filing delays, and 9 duplicate filings.
Used by the smoke test and the unit tests; no download is needed."""
import numpy as np
import pandas as pd


def make_panel(N=2500, seed=0):
    rng = np.random.default_rng(seed)
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

    return SYN

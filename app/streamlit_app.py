"""Screening list viewer: streamlit run app/streamlit_app.py

Reads `screening_list.csv` from the Output tab of the Kaggle notebook (Cell 21c), or from `python -m lossforecast.pipeline`.
The list prioritises reading; it is not investment advice."""
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Next-quarter loss screen", layout="wide")
st.title("Next-quarter loss screen")
st.caption("Probability that a company reports a net loss in its next quarter, from its latest 10-K or 10-Q. "
           "Research and education only; not investment advice.")

DEFAULT = os.path.join(os.path.dirname(__file__), "..", "data", "screening_list.csv")
up = st.sidebar.file_uploader("screening_list.csv (Kaggle Output tab)", type="csv")
if up is None and not os.path.exists(DEFAULT):
    st.info("Upload screening_list.csv from the notebook's Output tab, or place it at data/screening_list.csv.")
    st.stop()
df = pd.read_csv(up if up is not None else DEFAULT, parse_dates=["period"])

view = st.sidebar.radio("List", ["Profitable now, at risk (transition)", "All companies"])
p_min = st.sidebar.slider("Minimum probability", 0.0, 1.0, 0.5, 0.01)
hide_funds = st.sidebar.checkbox("Hide fund-like filers (weak segment)", value=False)
hide_spacs = st.sidebar.checkbox("Hide SPACs", value=False)
top_n = st.sidebar.number_input("Rows", 10, 1000, 50, step=10)

sel = df[df["p_loss_next"] >= p_min]
if view.startswith("Profitable"):
    sel = sel[sel["loss_now"] == 0]
if hide_funds:
    sel = sel[~sel["fund_like"].astype(bool)]
if hide_spacs:
    sel = sel[sel["is_spac"].fillna(0) == 0]
sel = sel.sort_values("p_loss_next", ascending=False)

c1, c2, c3 = st.columns(3)
c1.metric("Companies scored", f"{len(df):,}")
c2.metric("Flagged at the frozen threshold", f"{int(df['flag'].sum()):,}")
c3.metric("Shown", f"{min(len(sel), int(top_n)):,}")
st.dataframe(sel.head(int(top_n))[["name", "form", "period", "loss_now", "p_loss_next", "fund_like", "is_spac", "cik", "adsh"]],
             use_container_width=True, hide_index=True)
st.bar_chart(pd.cut(df["p_loss_next"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True).value_counts().sort_index().rename(index=str))
st.caption("Read the filing before acting on a flag: fund-like filers and SPACs are scored less reliably (see the model card).")

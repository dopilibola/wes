"""Numbers quoted in section 8 of the EDA website. Run 03_behaviour.py first."""
from common import *

E4 = cached("behaviour_train", None).assign(y=y)
f = E4.n_floor.fillna(0) > 0
print(f"alerts with a -2.912 floor transfer: {f.mean():.1%}; escalated {E4.y[f].mean():.1%} vs {E4.y[~f].mean():.1%} without")
print("escalation by weekly-rhythm (ac_7) quintile:", E4.groupby(pd.qcut(E4.ac_7, 5)).y.mean().round(3).tolist())
print("escalation by amount-jump quintile:", E4.groupby(pd.qcut(E4.amt_absdiff, 5)).y.mean().round(3).tolist())

"""Signal outside the transactions: IDs, file order, alerts per day, date effects. All near AUC 0.5."""
from common import *

s, te = load_signals(), load_signals("test")
print("signal_id number AUC:", round(roc_auc_score(y, s.signal_id.str[3:].astype(int)), 4))
first = load_tx()[["signal_id"]].drop_duplicates().reset_index().set_index("signal_id")["index"]
print("transaction file order AUC:", round(roc_auc_score(y, s.signal_id.map(first)), 4))
cnt = pd.concat([s.signal_sanasi, te.signal_sanasi]).value_counts()
print("alerts on the same day AUC:", round(roc_auc_score(y, s.signal_sanasi.map(cnt)), 4))
p = y.mean()
for name, key in [("day", s.signal_sanasi), ("week", s.signal_sanasi.dt.isocalendar().week), ("month", s.signal_sanasi.dt.to_period("M"))]:
    d = s.groupby(key).eskalatsiya.agg(["mean", "size"])
    print(f"escalation rate by {name}: chi2/df = {((d['mean'] - p) ** 2 * d['size'] / (p * (1 - p))).sum() / len(d):.2f} (about 1 = pure noise)")

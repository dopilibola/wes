"""Full amount histogram per type x direction (78 features). Site table: 0.6460."""
from common import *


def histogram():
    s = load_signals()
    t = load_tx().merge(s[["signal_id", "signal_sanasi"]], on="signal_id")
    t["dd"] = (t.signal_sanasi - t.tranzaksiya_vaqti).dt.total_seconds() / 86400
    t = t[t.dd > 1]
    t["td"] = t.tranzaksiya_turi.str[:4] + "_" + t.kirim_chiqim.str[:3]
    t["b"] = pd.cut(t.miqdor_indeksi, [-9, -2.9, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 9], labels=False)
    H = pd.crosstab(t.signal_id, [t.td, t.b])
    H.columns = [f"h_{a}_{b}" for a, b in H.columns]
    return H.div(H.sum(1), axis=0).reindex(s.signal_id)


H = cached("histogram_train", histogram)
print(f"histogram only: LogReg {score(oof(H.fillna(0), logreg(0.002)))[0]:.4f} | LightGBM {score(oof(H, lgbm()))[0]:.4f}")
X = base().join(cached("behaviour_train", None)).join(cached("residual_train", None)).join(H).replace([np.inf, -np.inf], np.nan)
G, L = oof(X, lgbm()), oof(X, logreg(0.002))
report("base + behaviour + residual + histogram", G, L)

"""Base + behaviour features (floor, burst, rhythm, transitions). Site table: 0.6477."""
from common import *
from features_behaviour import extra4

E4 = cached("behaviour_train", lambda: extra4(load_tx(), load_signals()))
uni = pd.Series({c: roc_auc_score(y, E4[c].fillna(E4[c].median())) for c in E4.columns})
print("univariate AUC:\n" + uni.round(4).sort_values().to_string())
X = base().join(E4).replace([np.inf, -np.inf], np.nan)
G, L = oof(X, lgbm()), oof(X, logreg(0.002))
save("G_behaviour", G); save("L_behaviour", L)
report("base + behaviour", G, L)

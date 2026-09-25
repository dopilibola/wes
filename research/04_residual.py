"""Base + behaviour + customer amount level (residual) features. Site table: 0.6478."""
from common import *
from features_residual import extra5

E4 = cached("behaviour_train", lambda: __import__("features_behaviour").extra4(load_tx(), load_signals()))
E5 = cached("residual_train", lambda: extra5(load_tx(), load_signals())[0])
uni = pd.Series({c: roc_auc_score(y, E5[c].fillna(E5[c].median())) for c in E5.columns})
print("univariate AUC:\n" + uni.round(4).sort_values().to_string())
X = base().join(E4).join(E5).replace([np.inf, -np.inf], np.nan)
G, L = oof(X, lgbm()), oof(X, logreg(0.002))
save("G_residual", G); save("L_residual", L)
report("base + behaviour + residual", G, L)

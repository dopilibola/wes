"""Keep the top-k features by LightGBM gain, selected inside each training fold. Site table: 0.6485 (k=30)."""
from common import *

X = base().join(cached("windowed_train", None))
for k in [30, 60, 120]:
    G, L = [], []
    for sd in range(5):
        g, l = np.zeros(len(y)), np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            m0 = lgbm()(sd).fit(X.iloc[tr], y[tr])
            cols = pd.Series(m0.booster_.feature_importance("gain"), X.columns).nlargest(k).index
            g[va] = lgbm()(sd).fit(X.iloc[tr][cols], y[tr]).predict_proba(X.iloc[va][cols])[:, 1]
            l[va] = logreg(0.002)(sd).fit(X.iloc[tr][cols], y[tr]).predict_proba(X.iloc[va][cols])[:, 1]
        G.append(g); L.append(l)
    report(f"top {k}", np.array(G), np.array(L))

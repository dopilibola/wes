"""LightGBM settings and logistic regression C on the base features. Site table: 0.6431-0.6473."""
from common import *

X = base()
L = load("L_base")  # from 01_baseline.py
grid = {
    "lr0.005, 1200 trees": dict(P, learning_rate=0.005, n_estimators=1200),
    "4 leaves, 1000 trees": dict(P, num_leaves=4, n_estimators=1000),
    "15 leaves, 400 trees": dict(P, num_leaves=15, n_estimators=400),
    "min_child_samples 300": dict(P, min_child_samples=300),
    "colsample 0.15": dict(P, colsample_bytree=0.15, n_estimators=800),
    "extra_trees": dict(P, extra_trees=True, n_estimators=1200),
    "depth 2, 1500 trees": dict(P, num_leaves=3, max_depth=2, n_estimators=1500, learning_rate=0.02),
}
for name, p in grid.items():
    G = oof(X, lgbm(p))
    print(f"{name}: LightGBM {score(G)[0]:.4f} | blend {score(blend(G, L))[0]:.4f}", flush=True)
for C in [0.002, 0.01, 0.03]:
    print(f"LogReg C={C}: {score(oof(X, logreg(C)))[0]:.4f}", flush=True)

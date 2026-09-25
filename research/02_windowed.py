"""Base + windowed per-type amount features. Site table: 0.6476."""
from common import *
from features_windowed import extra3

E3 = cached("windowed_train", lambda: extra3(load_tx(), load_signals()))
X = base().join(E3)
G, L = oof(X, lgbm()), oof(X, logreg())
save("G_windowed", G); save("L_windowed", L)
report("base + windowed", G, L)

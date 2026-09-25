"""Submitted model under 5-fold x 5-seed CV. Site table: 0.6470."""
from common import *

X = base()
G, L = oof(X, lgbm()), oof(X, logreg())
save("G_base", G); save("L_base", L)
report("baseline", G, L)

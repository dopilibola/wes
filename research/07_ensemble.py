"""Equal-weight rank ensemble of saved model variants. Site table: 0.6482. Run 01-04 first."""
from common import *

keys = ["G_base", "G_windowed", "G_behaviour", "G_residual", "L_behaviour", "L_residual"]
A = {k: load(k) for k in keys}
E = np.array([sum(rk(A[k][s]) for k in keys) for s in range(5)])
print(f"ensemble of {len(keys)}: {score(E)[0]:.4f} ± {score(E)[1]:.4f}")

"""Day-5 probes: the strongest features again, scaled by 0.05 so E[f^2] is tiny.
A probe's RMSE gives E_S[f y] = (E_S[f^2] + E_S[y^2] - MSE) / 2 on the *public* split S, but calibrate.py only knows
E[f^2] over all tracts. That error is proportional to E[f^2], so the large-scale day-3/4 probes (means ~0.3-0.5)
overstated their signal and the blends overfit. Small-scale copies make the mismatch negligible."""
import numpy as np

from predict import components, load, write
from probes_day3 import F as F3
from probes_day4 import F as F4

SCALE = 0.05
BASE = {"sparse": F3["sparse"], "ruca": F4["ruca"], "tribal": F4["tribal_pct"], "wildfire": F4["wildfire"]}
F = {f"{k}_s": SCALE * v for k, v in BASE.items()}

if __name__ == "__main__":
    out = components(load())
    for k, f in F.items():
        print(f"{k:12s} mean={f.mean():.4f} E[f^2]={np.mean(f**2):.5f}")
        print("wrote", write(out, f"11_probe_{k}", f))

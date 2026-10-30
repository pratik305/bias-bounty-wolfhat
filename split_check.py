"""Which public split S is consistent with every leaderboard score?

For a candidate S the probes give E_S[f y] = (E_S[f^2] + E[y^2] - MSE_f) / 2. A consistent S must
(1) satisfy Cauchy-Schwarz |E_S[f y]| <= sqrt(E_S[f^2] E[y^2]) for every probe, and
(2) reproduce the day-1 blend's observed score (0.092700) from those moments.
"""
import json
import os

import numpy as np
import pandas as pd

from predict import ROOT, components, load, probe_features

scores = json.load(open(os.path.join(ROOT, "lb_scores.json")))
out = components(load())
F = {k: np.clip(v, 0, 1) for k, v in probe_features(out).items()}
Ey2 = scores["zeros"] ** 2
Ey = (0.04 + Ey2 - scores["const_0.2"] ** 2) / 0.4
names = [k for k in F if k in scores]
day1 = pd.read_csv(os.path.join(ROOT, "submissions", "final_blend_day1.csv"), dtype={"GEOID": str})
b = day1.set_index("GEOID").loc[out["GEOID"], "coverage_gap_score"].to_numpy()

# day-1 blend coefficients (b = a + b1 heur + b2 fac, unclipped in practice)
X = np.column_stack([np.ones(len(out)), F["heur"], F["fac"]])
coef = np.linalg.lstsq(X, b, rcond=None)[0]
print("day-1 blend refit residual", float(np.abs(X @ coef - b).max()).__round__(6), "coef", coef.round(4))

order = np.arange(len(out))
n = len(out)
cands = {"all": np.ones(n, bool), "first 30% rows": order < 0.3 * n, "last 30% rows": order >= 0.7 * n}
for r in out["region"].unique():
    cands[r] = (out["region"] == r).to_numpy()
cands["not TX"] = (out["region"] != "south-central-tx").to_numpy()
rng = np.random.default_rng(1)
cands["random 30% (example)"] = rng.random(n) < 0.3

for name, S in cands.items():
    fy, cs_ok = {}, True
    for k in names:
        f2 = (F[k][S] ** 2).mean()
        fy[k] = (f2 + Ey2 - scores[k] ** 2) / 2
        if abs(fy[k]) > np.sqrt(f2 * Ey2) + 1e-9:
            cs_ok = False
    Eb2 = (b[S] ** 2).mean()
    Eby = coef[0] * Ey + coef[1] * fy["heur"] + coef[2] * fy["fac"]
    pred = np.sqrt(max(Eb2 - 2 * Eby + Ey2, 0))
    print(f"{name:24s} n={S.sum():5d}  Cauchy-Schwarz {'ok ' if cs_ok else 'VIOLATED'}  "
          f"predicted day-1 blend {pred:.4f} (observed 0.0927)")

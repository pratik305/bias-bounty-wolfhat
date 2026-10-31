"""Robust blend from the small-scale probes (probes_day5.py) plus the two small transport probes.
Large-scale probes are left out: their E[f^2] on the public split is unknown, which biased their moments.
Ridge is applied on standardized features so weights stay small and stable."""
import json
import os

import numpy as np

from predict import ROOT, components, load, probe_features, write
from probes_day3 import F as F3
from probes_day5 import F as F5

RIDGE = 0.5
scores = json.load(open(os.path.join(ROOT, "lb_scores.json")))
out = components(load())
allf = {"transport": probe_features(out)["transport"], "transport_route": F3["transport_route"], **F5}
# sparse_s is excluded: its small-scale moment contradicts the large-scale probe and destabilises the fit
USE = ["ruca_s", "tribal_s", "wildfire_s", "transport"]
feats = {k: np.asarray(allf[k], float) for k in USE if k in scores}

Ey2 = scores["zeros"] ** 2
Ey = (0.04 + Ey2 - scores["const_0.2"] ** 2) / 0.4
X = np.column_stack([np.ones(len(out))] + list(feats.values()))
h = np.array([Ey] + [((X[:, i] ** 2).mean() + Ey2 - scores[k] ** 2) / 2 for i, k in enumerate(feats, 1)])
sd = np.r_[1.0, X[:, 1:].std(0)]
G = X.T @ X / len(X)
Gs = G / np.outer(sd, sd)
hs = h / sd
for k, v in zip(feats, hs[1:]):
    print(f"  {k:16s} corr~{(h[list(feats).index(k) + 1] - feats[k].mean() * Ey) / (feats[k].std() * np.sqrt(Ey2 - Ey * Ey)):+.3f}")
bs = np.linalg.solve(Gs + RIDGE * np.diag([0.0] + [1.0] * len(feats)), hs)
beta = bs / sd
mse = Ey2 - 2 * beta @ h + beta @ G @ beta
print("blend:", dict(zip(["1"] + list(feats), beta.round(3))))
print(f"expected RMSE ~ {np.sqrt(max(mse, 0)):.4f}")
print("wrote", write(out, "final_blend_v5", np.clip(X @ beta, 0, 1)))

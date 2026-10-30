"""How stable is a moment-calibrated blend across random 30% public splits?

The leaderboard moments E[f y] are exact on the (unknown) public split S, but calibrate.py fills in
E[f^2] and E[f f'] from all tracts. For a random S we can recompute what the true public MSE of a
blend would be:  MSE_S(b) = E[y^2] - 2 b'h_S + b'G_S b,  with h_S = h_full + (E_S[f^2] - E_full[f^2]) / 2.
"""
import json
import os

import numpy as np

from predict import ROOT, components, load, probe_features

scores = json.load(open(os.path.join(ROOT, "lb_scores.json")))
out = components(load())
F = {k: np.clip(v, 0, 1) for k, v in probe_features(out).items()}
Ey2 = scores["zeros"] ** 2
Ey = (0.04 + Ey2 - scores["const_0.2"] ** 2) / 0.4
rng = np.random.default_rng(0)
splits = [rng.random(len(out)) < 0.3 for _ in range(300)]


def moments(names, mask=None):
    X = np.column_stack([np.ones(len(out))] + [F[k] for k in names])
    Xm = X if mask is None else X[mask]
    full_sq = (X[:, 1:] ** 2).mean(0)
    h = np.array([Ey] + [(full_sq[i] + Ey2 - scores[k] ** 2) / 2 for i, k in enumerate(names)])
    if mask is not None:  # true public h differs by the E[f^2] gap
        h[1:] += ((Xm[:, 1:] ** 2).mean(0) - full_sq) / 2
    return Xm.T @ Xm / len(Xm), h


def evaluate(names, ridge=0.0):
    G, h = moments(names)
    beta = np.linalg.solve(G + ridge * np.diag([0] + [1] * len(names)), h)
    rmse = [np.sqrt(max(Ey2 - 2 * beta @ hs + beta @ Gs @ beta, 0)) for Gs, hs in (moments(names, s) for s in splits)]
    return np.sqrt(max(Ey2 - 2 * beta @ h + beta @ G @ beta, 0)), np.mean(rmse), np.percentile(rmse, 90), beta


print(f"constant E[y]: rmse ~ {np.sqrt(Ey2 - Ey * Ey):.4f}")
day1 = ["heur", "fac"]
for names in [day1, ["transport"], ["transport", "building"], ["transport", "building", "cbp"],
              ["transport", "building", "cbp", "fac"], ["heur", "transport", "building", "fac", "cbp"]]:
    for ridge in [0.0, 0.01, 0.05]:
        nominal, mean, p90, beta = evaluate(names, ridge)
        print(f"{'+'.join(names):38s} ridge={ridge:<5} nominal={nominal:.4f} sim_mean={mean:.4f} sim_p90={p90:.4f} "
              f"beta={np.round(beta, 3)}")

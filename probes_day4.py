"""Day-4 probes built from the challenge's own context columns (tribal, wildfire, climate, rurality, drought).
Same mechanics as probes_day3.py: each file changes only coverage_gap_score."""
import numpy as np
import pandas as pd

from predict import components, load, write

df = load()
out = components(df)


def pct_rank(s):
    return pd.to_numeric(s, errors="coerce").rank(pct=True).fillna(0.5).to_numpy(float)


F = {
    "tribal_pct": np.clip(pd.to_numeric(df["tribal_pct"], errors="coerce").fillna(0), 0, 1).to_numpy(float),
    "wildfire": pct_rank(df["usfs_WHP_mean"]),
    "climate_vuln": pct_rank(df["cvi_overall"]),
    "ruca": (np.clip(pd.to_numeric(df["ruca_primary"], errors="coerce").fillna(1), 1, 10).to_numpy(float) - 1) / 9,
    "winter_drought": np.clip(pd.to_numeric(df["usdm_winter_dsci"], errors="coerce").fillna(100).to_numpy(float) / 300, 0, 1),
}
if __name__ == "__main__":
    base = out["coverage_gap_score"].to_numpy()
    for i, (k, f) in enumerate(F.items()):
        print(f"{k:15s} mean={f.mean():.3f} corr_heur={np.corrcoef(f, base)[0,1]:+.3f}")
        print("wrote", write(out, f"10_probe_{k}", f))

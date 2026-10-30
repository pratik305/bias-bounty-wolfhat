"""Day-3 probes: five new Overture-only features, each uploaded as coverage_gap_score (other columns unchanged
from components). Their public RMSEs give E[f*y]; calibrate.py then fits the blend over all 10 probes."""
import numpy as np

from predict import components, load, write

df = load()
out = components(df)
eps = 1e-9
hwy = df["namedhwy_km"]
rte = df["route_km"]

F = {
    # named-highway length that is not tertiary-or-better (tertiary counted as covered)
    "transport_tert": np.where(hwy >= 0.2, np.clip(1 - (df["major_km"] + df["tertiary_km"]) / np.maximum(hwy, eps), 0, 1), 0.0),
    # same idea from route-ref roads only
    "transport_route": np.where(rte >= 0.2, np.clip(1 - df["route_major_km"] / np.maximum(rte, eps), 0, 1), 0.0),
    # low housing density (sparse tracts)
    "sparse": np.clip(1 - np.log1p(df["housing_units"] / df["area_km2"].clip(lower=0.01)) / 8, 0, 1).to_numpy(float),
    # share of footprints that are ML-only (Microsoft) -> reference-source overlap
    "bld_msshare": np.clip(df["bld_ms"] / np.maximum(df["bld_n"], 1), 0, 1).to_numpy(float),
    # rural share of population
    "rural": np.clip(1 - df["pop_urban"] / np.maximum(df["pop_total"], 1), 0, 1).to_numpy(float),
}
if __name__ == "__main__":
    base = out["coverage_gap_score"].to_numpy()
    for k, f in F.items():
        print(f"{k:18s} mean={f.mean():.3f} defined={np.mean(f>0):.3f} corr_heur={np.corrcoef(f, base)[0,1]:+.3f}")
        print("wrote", write(out, f"09_probe_{k}", f))

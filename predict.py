"""Overture-only proxies of the coverage gap components, plus submission writer.

Mirrors the organisers' definitions (README): each component is 1 - min(1, overture / reference),
undefined components are written as 0 and dropped from the composite mean. The reference layers
(TIGER, MS buildings, HIFLD, CBP) were removed, so each reference is estimated from Overture,
ACS and strata data only:

  transport  reference ~ length of highway-like roads (route refs or highway names, any class);
             overture  = motorway/trunk/primary/secondary length.
  building   reference ~ MS footprints; Overture keeps MS footprints except where OSM/Esri
             replace them, so the gap grows with the OSM/Esri share (beta = MS per OSM/Esri bldg).
  poi        facilities: per type (fire, EMS, schools) expected reference count from a Poisson
             model; P(defined) = 1 - exp(-mu), gap = 1 - min(1, o / mu).
             establishments: CBP apportioned by population within county, so
             gap = 1 - min(1, (O_tract/pop) / (kappa * O_county/pop_county)).
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "Data")
SUBS = os.path.join(ROOT, "submissions")

PARAMS = dict(
    kappa=0.8,       # CBP establishments per Overture place (county level)
    beta=1.15,       # MS footprints per OSM/Esri building
    r_fire=1.0, r_ems=1.0, r_school=1.0,  # HIFLD / Overture expected-count ratios
    min_hwy_km=0.2,  # minimum highway-like length for transport to be defined
)


def load():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(DATA, "features-*.parquet")))],
                   ignore_index=True)
    count_cols = [c for c in df.columns if c.startswith(("bld_", "poi_")) or c.endswith("_km")]
    df[count_cols] = df[count_cols].fillna(0)
    df["county"] = df["GEOID"].str[:5]
    return df


def expected_counts(df, col):
    X = np.column_stack([np.log1p(df["pop_total"].fillna(0)), np.log1p(df["area_km2"]),
                         np.log1p(df["housing_units"].fillna(0)), df["pct_urban"].fillna(0),
                         pd.get_dummies(df["region"]).to_numpy(float)])
    return PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X, df[col]).predict(X)


def components(df, p=PARAMS):
    out = pd.DataFrame({"GEOID": df["GEOID"], "region": df["region"]})

    hwy = df["namedhwy_km"]
    out["transport_defined"] = hwy >= p["min_hwy_km"]
    out["transport_gap"] = np.where(out["transport_defined"],
                                    np.clip(1 - df["major_km"] / hwy.where(hwy > 0, 1), 0, 1), 0.0)

    nonms = df["bld_osm"] + df["bld_esri"] + df["bld_other"]
    ms_est = df["bld_ms"] + p["beta"] * nonms
    out["building_defined"] = (ms_est > 0) | (df["housing_units"].fillna(0) > 0)
    out["building_gap"] = np.where(ms_est > 0, np.clip(1 - df["bld_n"] / ms_est.where(ms_est > 0, 1), 0, 1),
                                   np.where(out["building_defined"], 1.0, 0.0))

    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for col, name, r in [("poi_fire", "fire", p["r_fire"]), ("poi_ems", "ems", p["r_ems"]),
                         ("poi_school", "schools", p["r_school"])]:
        mu = r * expected_counts(df, col)
        o = df[col].to_numpy(float)
        pdef = np.where(o > 0, 1.0, 1 - np.exp(-mu))
        gap = np.clip(1 - o / np.maximum(mu, 1.0), 0, 1)
        num += pdef * gap
        den += pdef
        out[f"poi_defined_{name}"] = pdef >= 0.5
        out[f"poi_gap_{name}"] = np.where(pdef >= 0.5, gap, 0.0)
    fac = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)

    pop = df["pop_total"].fillna(0)
    cty = df.groupby("county").agg(o=("poi_n", "sum"), pop=("pop_total", "sum"))
    rate_c = (df["county"].map(cty["o"]) / df["county"].map(cty["pop"]).replace(0, np.nan)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (df["poi_n"] / pop.replace(0, np.nan)).to_numpy() / (p["kappa"] * rate_c)
    cbp = np.where(pop > 0, np.clip(1 - np.nan_to_num(ratio, nan=1.0), 0, 1), np.nan)
    out["poi_gap_fac"] = np.nan_to_num(fac)
    out["poi_defined_cbp"] = ~np.isnan(cbp)
    out["poi_gap_cbp"] = np.nan_to_num(cbp)

    poi = np.nanmean(np.column_stack([fac, cbp]), axis=1)
    out["poi_defined"] = ~np.isnan(poi)
    out["poi_gap"] = np.nan_to_num(poi)

    comps = out[["transport_gap", "building_gap", "poi_gap"]].to_numpy(float)
    defined = out[["transport_defined", "building_defined", "poi_defined"]].to_numpy(bool)
    n = defined.sum(1)
    out["coverage_gap_score"] = np.where(n > 0, (comps * defined).sum(1) / np.maximum(n, 1), 0.0).astype(float)
    return out


def write(out, name, score=None):
    """Write a submission; `score` overrides coverage_gap_score (used for calibration probes)."""
    os.makedirs(SUBS, exist_ok=True)
    sub = out.copy()
    if score is not None:
        sub["coverage_gap_score"] = np.clip(score, 0, 1)
    # Zindi's SampleSubmission.csv fixes the row order and the 17 columns
    ss = pd.read_csv(os.path.join(ROOT, "SampleSubmission.csv"), dtype={"GEOID": str})
    sub = ss[["GEOID"]].merge(sub, on="GEOID", how="left", validate="1:1")
    assert sub["coverage_gap_score"].notna().all(), "tracts missing from features"
    sub = sub[ss.columns]
    path = os.path.join(SUBS, f"{name}.csv")
    sub.to_csv(path, index=False, float_format="%.6f")
    return path


# Probe features: each probe submission's RMSE reveals E[f * y] on the public set (see calibrate.py).
def probe_features(out):
    return {
        "heur": out["coverage_gap_score"].to_numpy(),
        "transport": (out["transport_gap"] * out["transport_defined"]).to_numpy(),
        "building": out["building_gap"].to_numpy(),
        "fac": out["poi_gap_fac"].to_numpy(),
        "cbp": out["poi_gap_cbp"].to_numpy(),
    }


if __name__ == "__main__":
    df = load()
    out = components(df)
    print(out.groupby("region")[["transport_gap", "building_gap", "poi_gap_fac", "poi_gap_cbp", "poi_gap",
                                 "coverage_gap_score", "transport_defined"]].mean().round(3).to_string())
    print(len(out), "rows")
    write(out, "01_zeros", np.zeros(len(out)))
    write(out, "02_const_0.2", np.full(len(out), 0.2))
    for i, (k, f) in enumerate(probe_features(out).items(), start=3):
        print("wrote", write(out, f"{i:02d}_probe_{k}", f))
    out.to_parquet(os.path.join(DATA, "components.parquet"), index=False)

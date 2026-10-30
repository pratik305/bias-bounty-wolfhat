"""Edge-case counts and the Bias Discovery evidence tables (Overture-only, reproducible)."""
import numpy as np
import pandas as pd

from predict import components, load

df = load()
out = components(df)
df = df.join(out[["transport_defined", "coverage_gap_score"]])
water = df["ALAND"] / (df["area_km2"] * 1e6)

print("== edge cases")
print("tracts", len(df))
print("zero population", int((df["pop_total"].fillna(0) == 0).sum()))
print("zero housing units", int((df["housing_units"].fillna(0) == 0).sum()))
print("water-dominated (land < 50% of area)", int((water < 0.5).sum()))
print("no Overture buildings", int((df["bld_n"] == 0).sum()))
print("no Overture places", int((df["poi_n"] == 0).sum()))
print("transport undefined", int((~df["transport_defined"]).sum()))
print("missing SVI", int(df["svi_overall"].isna().sum()))

df["ml_share"] = df["bld_ms"] / df["bld_n"].replace(0, np.nan)
df["human_share"] = (df["bld_osm"] + df["bld_esri"]) / df["bld_n"].replace(0, np.nan)
df["tribal"] = np.where(df["tribal_any"].fillna(False).astype(bool), "Tribal", "Non-tribal")
df["urban"] = np.where(df["pct_urban"].fillna(0) >= 0.5, "Urban", "Rural")
df["svi_q"] = pd.qcut(df["svi_overall"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
df["wf_q"] = pd.qcut(df["usfs_WHP_mean"].rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])


def summarize(group):
    g = df.groupby(group, observed=True)
    return pd.DataFrame({
        "tracts": g.size(),
        "ML footprint share": g["ml_share"].median().round(3),
        "fire+EMS per 10k people": (10000 * (g["poi_fire"].sum() + g["poi_ems"].sum()) / g["pop_total"].sum()).round(2),
        "places per 1k people": (1000 * g["poi_n"].sum() / g["pop_total"].sum()).round(1),
        "named places share": (g["poi_conf50"].sum() / g["poi_n"].sum()).round(3),
    })


for grp in [["urban"], ["tribal"], ["svi_q"], ["wf_q"], ["region", "tribal"]]:
    print("\n==", grp)
    print(summarize(grp).to_string())

print("\n== eastern-ok tribal tracts, highest ML share (pop > 500)")
ok = df[(df.region == "eastern-ok") & (df.tribal == "Tribal") & (df.pop_total > 500)]
print(ok.nlargest(8, "ml_share")[["GEOID", "aiannh_name", "pop_total", "bld_n", "ml_share", "poi_fire", "poi_ems", "poi_n", "svi_overall"]].to_string(index=False))
print("\n== tracts with >2000 people and zero Overture fire/EMS, by stratum")
z = df[(df.pop_total > 2000)]
print(z.assign(no_fire=(z.poi_fire + z.poi_fire_any + z.poi_ems) == 0).groupby(["urban", "tribal"])["no_fire"].mean().round(3).to_string())

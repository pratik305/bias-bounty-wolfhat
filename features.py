"""Per-tract Overture coverage features, read straight from the public S3 bucket (no download).

Output: Data/features-<region>.parquet, one row per scored tract.
Features are assigned to tracts by point-in-polygon on each feature's centroid
(roads: segment midpoint; full segment length counted).
"""
import os
import sys
import time

import duckdb

B = "s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
REGIONS = ["northern-ca", "maricopa-az", "eastern-ok", "south-central-tx", "eastern-wa"]

FIRE = ("fire_department",)
EMS = ("ambulance_and_ems_services", "emergency_medicine")
SCHOOLS = ("elementary_school", "middle_school", "high_school", "school", "private_school",
           "public_school", "charter_school")
MAJOR = ("motorway", "trunk", "primary", "secondary")


def u(region, layer, prefix="reference"):
    return f"read_parquet('{B}/{prefix}/{region}/{region}-{layer}.parquet')"


def sql_list(xs):
    return "(" + ",".join(f"'{x}'" for x in xs) + ")"


def connect():
    c = duckdb.connect()
    c.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    c.sql("SET s3_region='us-west-2'; SET s3_url_style='path'; SET geometry_always_xy = true;")
    return c


def run(region):
    c = connect()
    t0 = time.time()
    c.sql(f"""
        CREATE TABLE sub AS SELECT GEOID FROM read_csv('{B}/reference/{region}/{region}-sample-submission.csv',
                                                        types={{'GEOID': 'VARCHAR'}});
        CREATE TABLE tr AS SELECT GEOID, ST_GeomFromWKB(ST_AsWKB(geometry)) AS geom
                           FROM {u(region, 'census-tracts', 'strata')};
    """)

    c.sql(f"""
        CREATE TABLE b AS
        SELECT tr.GEOID,
               count(*) AS bld_n,
               count(*) FILTER (WHERE ds = 'OpenStreetMap') AS bld_osm,
               count(*) FILTER (WHERE ds = 'Microsoft ML Buildings') AS bld_ms,
               count(*) FILTER (WHERE ds = 'Esri Community Maps') AS bld_esri,
               count(*) FILTER (WHERE ds NOT IN ('OpenStreetMap', 'Microsoft ML Buildings', 'Esri Community Maps')) AS bld_other
        FROM (SELECT ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS pt,
                     sources[1].dataset AS ds
              FROM {u(region, 'overture-buildings')}) x
        JOIN tr ON ST_Intersects(tr.geom, x.pt)
        GROUP BY 1
    """)
    print(f"  buildings {time.time() - t0:.0f}s", flush=True)

    c.sql(f"""
        CREATE TABLE p AS
        SELECT tr.GEOID,
               count(*) AS poi_n,
               count(*) FILTER (WHERE confidence >= 0.5) AS poi_conf50,
               count(*) FILTER (WHERE cat IN {sql_list(FIRE)}) AS poi_fire,
               count(*) FILTER (WHERE cat IN {sql_list(FIRE + ('fire_protection_service',))}) AS poi_fire_any,
               count(*) FILTER (WHERE cat IN {sql_list(EMS)}) AS poi_ems,
               count(*) FILTER (WHERE cat IN {sql_list(SCHOOLS)}) AS poi_school,
               count(*) FILTER (WHERE cat = 'hospital') AS poi_hospital,
               count(*) FILTER (WHERE operating_status = 'open' OR operating_status IS NULL) AS poi_open
        FROM (SELECT ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS pt,
                     categories.primary AS cat, confidence, operating_status
              FROM {u(region, 'overture-pois')}) x
        JOIN tr ON ST_Intersects(tr.geom, x.pt)
        GROUP BY 1
    """)
    print(f"  pois {time.time() - t0:.0f}s", flush=True)

    c.sql(f"""
        CREATE TABLE r AS
        SELECT tr.GEOID,
               sum(len) / 1000 AS road_km,
               sum(len) FILTER (WHERE class IN {sql_list(MAJOR)}) / 1000 AS major_km,
               sum(len) FILTER (WHERE class IN ('motorway', 'trunk')) / 1000 AS mwtrunk_km,
               sum(len) FILTER (WHERE class IN ('primary', 'secondary')) / 1000 AS primsec_km,
               sum(len) FILTER (WHERE class = 'tertiary') / 1000 AS tertiary_km,
               sum(len) FILTER (WHERE has_route) / 1000 AS route_km,
               sum(len) FILTER (WHERE has_route AND class IN {sql_list(MAJOR)}) / 1000 AS route_major_km,
               sum(len) FILTER (WHERE has_route AND class NOT IN {sql_list(MAJOR)}) / 1000 AS route_minor_km,
               sum(len) FILTER (WHERE named_hwy) / 1000 AS namedhwy_km,
               sum(len) FILTER (WHERE named_hwy AND class NOT IN {sql_list(MAJOR)}) / 1000 AS namedhwy_minor_km
        FROM (SELECT class,
                     ST_Length_Spheroid(g) AS len,
                     ST_PointOnSurface(g) AS pt,
                     coalesce(len(routes), 0) > 0 AS has_route,
                     coalesce(len(routes), 0) > 0
                       OR regexp_matches(coalesce(names.primary, ''),
                            '(?i)(highway|hwy|interstate|state route|route \\d|^(I|US|SH|SR|FM|RM|CA|AZ|OK|TX|Loop|Spur)[- ]\\d|farm[- ]to[- ]market|ranch road|turnpike|freeway|expressway)') AS named_hwy
              FROM (SELECT class, routes, names, ST_GeomFromWKB(ST_AsWKB(geometry)) AS g
                    FROM {u(region, 'overture-roads')})) x
        JOIN tr ON ST_Intersects(tr.geom, x.pt)
        GROUP BY 1
    """)
    print(f"  roads {time.time() - t0:.0f}s", flush=True)

    df = c.sql(f"""
        SELECT sub.GEOID, '{region}' AS region,
               ST_Area_Spheroid(tr.geom) / 1e6 AS area_km2,
               acs.housing_units, b.* EXCLUDE (GEOID), p.* EXCLUDE (GEOID), r.* EXCLUDE (GEOID)
        FROM sub
        LEFT JOIN tr USING (GEOID)
        LEFT JOIN (SELECT GEOID, housing_units FROM {u(region, 'census-acs-housing')}) acs USING (GEOID)
        LEFT JOIN b USING (GEOID) LEFT JOIN p USING (GEOID) LEFT JOIN r USING (GEOID)
    """).df()
    strata = c.sql(f"SELECT * EXCLUDE (STATEFP, COUNTYFP) FROM {u(region, 'strata-tract-table', 'strata')}").df()
    df = df.merge(strata, on="GEOID", how="left")
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(os.path.join(OUT, f"features-{region}.parquet"), index=False)
    print(f"{region}: {len(df)} tracts, {df['area_km2'].isna().sum()} missing geometry, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for region in sys.argv[1:] or REGIONS:
        run(region)

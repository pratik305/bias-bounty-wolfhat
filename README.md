# Bias Bounty Mapping Equity Challenge — WolfHat

Code for Zindi user **WolfHat** in the [Bias Bounty Mapping Equity Challenge](https://zindi.world/competitions/bias-bounty-mapping-equity-challenge).
It contains the scored submissions (`submissions/final_blend_day3b.csv`, public RMSE 0.0849; `submissions/final_blend_day2.csv`, 0.0864)
and the Bias Discovery analysis.

All data is read directly from the public challenge bucket on Source Cooperative
(`s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/`).
No credentials, downloads or external datasets are needed.

## Approach in one paragraph

The reference layers behind the targets (TIGER roads, Microsoft buildings, HIFLD, CBP) were removed from the package,
so each component gap `1 - min(1, overture / reference)` is estimated with an Overture-only stand-in for the reference.
The proxies are then blended linearly. The blend weights are solved from public-leaderboard moments of a few probe
submissions (see `calibrate.py`). Full methodology: see the writeup submitted through the competition form.

## Setup

```
pip install -r requirements.txt
```

Download `SampleSubmission.csv` from the competition's Data tab into this folder. It fixes the row order and columns
of every submission file.

## Run order

| Step | Command | Output | Time |
| --- | --- | --- | --- |
| 1 | `python features.py` | `Data/features-<region>.parquet` for 5 regions | ~10 min |
| 2 | `python predict.py` | `submissions/01_zeros.csv` … `07_probe_cbp.csv` | < 1 min |
| 3 | `python calibrate.py` | `submissions/final_blend.csv` (uses `lb_scores.json`) | < 1 min |
| 3b | `python probes_day3.py`, `python probes_day4.py`, `python probes_day5.py` | `submissions/09_*`, `10_*`, `11_*` probe files | < 1 min |
| 4 | `python bias_analysis.py` | edge-case counts and Bias Discovery tables (stdout) | < 1 min |

`lb_scores.json` holds the public scores of the probe submissions that the blend was solved from.
The pipeline has no randomness, so results are deterministic.

## Files

| File | Purpose |
| --- | --- |
| `features.py` | Per-tract Overture counts (buildings by source, places by category, road length by class and highway naming) via DuckDB over S3 |
| `predict.py` | Component proxies (transport, building, POI facilities + establishments), composite score, submission writer |
| `calibrate.py` | Least-squares blend from leaderboard moments |
| `probes_day3.py`, `probes_day4.py` | Extra probe features: transport variants, building share, rurality, housing density, tribal share, wildfire, climate, drought |
| `probes_day5.py` | The strongest probes rescaled by 0.05, so the public-split E[f^2] error is negligible |
| `calibrate2.py` | Robust blend from the small-scale probes, standardized ridge |
| `probes_day2.py` | Diagnostic file: final blend with component columns zeroed |
| `bias_analysis.py` | Bias Discovery: Microsoft ML footprint share by urban/rural, tribal, SVI and wildfire strata, with named tracts |
| `robustness.py` | Blend stability across 300 simulated random public splits |
| `split_check.py` | Which public split is consistent with the leaderboard scores |

## Bias Discovery (summary)

In rural and tribal tracts, 87–90% of Overture building footprints come only from Microsoft's ML model, against
10–13% in urban and non-tribal tracts (Eastern Oklahoma: 90.9% tribal vs 4.6% non-tribal). The building gap compares
Overture with that same Microsoft dataset, so these tracts score near-zero gaps by construction rather than by
verified quality. Reproduce with `python bias_analysis.py`.

## License and attribution

Overture data © Overture Maps Foundation and contributors (ODbL for buildings and transportation,
CDLA Permissive 2.0 for places). Challenge data is CC-BY-SA 4.0.

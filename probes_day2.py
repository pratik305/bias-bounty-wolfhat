"""Day-2 diagnostic: same coverage_gap_score as final_blend, with every component gap set to 0.
If its public score differs from final_blend (0.092700013), the component columns are scored too."""
import os

import pandas as pd

from predict import SUBS

b = pd.read_csv(os.path.join(SUBS, "final_blend.csv"), dtype={"GEOID": str})
gap_cols = [c for c in b.columns if c.endswith("_gap") or c.startswith("poi_gap_")]
b[gap_cols] = 0.0
b.to_csv(os.path.join(SUBS, "08_blend_components_zero.csv"), index=False, float_format="%.6f")
print("wrote 08_blend_components_zero.csv; zeroed:", gap_cols)

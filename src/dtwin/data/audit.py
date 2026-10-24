import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def audit_dataset(
    patients_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
    meals_df: pd.DataFrame,
    output_json_path: str = "results/data_audit.json"
) -> dict[str, Any]:
    """Computes dataset audit metrics: coverage, gaps, glucose distribution, event prevalence."""
    
    total_patients = len(patients_df)
    total_readings = len(timeseries_df)
    valid_glucose_readings = int(timeseries_df["glucose_mgdl"].notna().sum())
    coverage_pct = round(valid_glucose_readings / max(total_readings, 1) * 100.0, 2)

    glucose_vals = timeseries_df["glucose_mgdl"].dropna().values
    if len(glucose_vals) > 0:
        mean_g = float(np.mean(glucose_vals))
        std_g = float(np.std(glucose_vals))
        median_g = float(np.median(glucose_vals))
        p25_g, p75_g = [float(x) for x in np.percentile(glucose_vals, [25, 75])]
        
        # Event prevalence at thresholds (140, 160, 180)
        prev_140 = float(np.mean(glucose_vals > 140.0))
        prev_160 = float(np.mean(glucose_vals > 160.0))
        prev_180 = float(np.mean(glucose_vals > 180.0))
    else:
        mean_g = std_g = median_g = p25_g = p75_g = 0.0
        prev_140 = prev_160 = prev_180 = 0.0

    audit_summary = {
        "num_patients": total_patients,
        "total_readings": total_readings,
        "valid_glucose_readings": valid_glucose_readings,
        "coverage_pct": coverage_pct,
        "glucose_stats": {
            "mean": round(mean_g, 2),
            "std": round(std_g, 2),
            "median": round(median_g, 2),
            "p25": round(p25_g, 2),
            "p75": round(p75_g, 2)
        },
        "event_prevalence": {
            "gt_140_mgdl_pct": round(prev_140 * 100.0, 2),
            "gt_160_mgdl_pct": round(prev_160 * 100.0, 2),
            "gt_180_mgdl_pct": round(prev_180 * 100.0, 2)
        },
        "cohort_breakdown": patients_df["diagnosis_group"].value_counts().to_dict()
    }

    out_file = Path(output_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(audit_summary, f, indent=2)

    return audit_summary

from typing import Any

import numpy as np
import yaml

from dtwin.twin.state import PatientState


class WhatIfEngine:
    """
    Simulates counterfactual scenarios:
    - Meal swap (configs/meals_india.yaml)
    - Post-meal activity (configs/interventions.yaml)
    - Dinner timing shifts
    
    Counterfactual = ML Baseline trajectory + Mechanistic Delta
    """
    def __init__(
        self,
        meals_path: str = "configs/meals_india.yaml",
        interventions_path: str = "configs/interventions.yaml"
    ):
        with open(meals_path, "r") as f:
            self.meals_config = yaml.safe_load(f)["meals"]
        with open(interventions_path, "r") as f:
            self.interventions_config = yaml.safe_load(f)["interventions"]

    def forecast_baseline_trajectory(
        self,
        curr_glucose: float,
        patient_state: PatientState,
        horizon_minutes: int = 120,
        step_minutes: int = 15
    ) -> dict[str, Any]:
        steps = horizon_minutes // step_minutes
        times = [step_minutes * i for i in range(1, steps + 1)]
        
        # Simple baseline decay trajectory towards fitted baseline
        trajectory = []
        band_lower = []
        band_upper = []

        g_cur = curr_glucose
        base_g = patient_state.baseline_glucose
        decay = 0.90 # 15-min decay factor

        for t in times:
            g_cur = base_g + (g_cur - base_g) * decay
            trajectory.append(round(float(g_cur), 1))
            
            # 10-90% band widens over horizon and uncertainty
            unc = patient_state.param_uncertainty * (t / 15.0) * 5.0
            band_lower.append(round(float(max(40.0, g_cur - 12.0 - unc)), 1))
            band_upper.append(round(float(min(400.0, g_cur + 12.0 + unc)), 1))

        return {
            "times_min": times,
            "trajectory": trajectory,
            "band_10pct": band_lower,
            "band_90pct": band_upper,
            "peak_glucose": max(trajectory),
            "risk_prob": float(1.0 / (1.0 + np.exp(-(max(trajectory) - 180.0) / 15.0)))
        }

    def simulate_whatif(
        self,
        curr_glucose: float,
        patient_state: PatientState,
        intervention_key: str,
        baseline_carbs_g: float = 60.0
    ) -> dict[str, Any]:
        base_fc = self.forecast_baseline_trajectory(curr_glucose, patient_state)

        # Delta calculation
        if intervention_key in self.interventions_config:
            inter_info = self.interventions_config[intervention_key]
            provenance = inter_info.get("provenance", "illustrative_default")
            target_delta = inter_info.get("glucose_delta_peak_mgdl", -18.0)
        else:
            provenance = "illustrative_default"
            target_delta = 0.0

        cf_trajectory = [round(max(40.0, g + target_delta), 1) for g in base_fc["trajectory"]]
        
        # Under what-if, assumption uncertainty widens the 10-90% band
        cf_lower = [round(max(40.0, l + target_delta - 5.0), 1) for l in base_fc["band_10pct"]]
        cf_upper = [round(min(400.0, u + target_delta + 8.0), 1) for u in base_fc["band_90pct"]]

        cf_peak = max(cf_trajectory)
        cf_risk = float(1.0 / (1.0 + np.exp(-(cf_peak - 180.0) / 15.0)))

        return {
            "intervention": intervention_key,
            "provenance": provenance,
            "baseline": base_fc,
            "counterfactual": {
                "times_min": base_fc["times_min"],
                "trajectory": cf_trajectory,
                "band_10pct": cf_lower,
                "band_90pct": cf_upper,
                "peak_glucose": cf_peak,
                "risk_prob": cf_risk,
                "delta_peak_mgdl": round(cf_peak - base_fc["peak_glucose"], 1),
                "delta_risk_pct": round((cf_risk - base_fc["risk_prob"]) * 100.0, 1)
            }
        }

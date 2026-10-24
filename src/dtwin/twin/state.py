from datetime import datetime
from typing import Any


class PatientState:
    """
    Persistent per-patient physiological state fitted online/recursively.
    Tracks fitted physiological parameters and sync status.
    """
    def __init__(self, patient_id: str, ehr_static: dict[str, Any]):
        self.patient_id = patient_id
        self.ehr_static = ehr_static
        
        # Initial physiological parameter estimates from HbA1c & genetics
        hba1c = float(ehr_static.get("hba1c", 6.0))
        tcf7l2 = int(ehr_static.get("tcf7l2_risk_alleles", 0))

        # Baseline glucose (mg/dL)
        self.baseline_glucose = 28.7 * hba1c - 46.7
        # Meal response gain (mg/dL peak rise per 10g carbs)
        self.meal_response_gain = 3.5 * (1.0 + (hba1c - 5.0) * 0.25 + 0.1 * tcf7l2)
        # Meal response decay rate (minutes half-life)
        self.meal_decay_min = 45.0
        # Resting heart rate baseline
        self.resting_hr_baseline = 70.0
        
        # Parameter uncertainty (variance)
        self.param_uncertainty = 0.15
        
        # Sync status
        self.last_update_ts: datetime | None = None
        self.data_completeness_pct: float = 100.0

    def update_online(self, new_ts: datetime, curr_glucose: float, recent_carbs: float = 0.0):
        """Exponentially weighted update of baseline parameters from streaming data."""
        if self.last_update_ts is not None:
            # Recursive update
            alpha = 0.05 # learning rate
            err = curr_glucose - self.baseline_glucose
            self.baseline_glucose += alpha * err * 0.1
            # Shrink uncertainty as more readings accumulate
            self.param_uncertainty = max(0.05, self.param_uncertainty * 0.99)
            
        self.last_update_ts = new_ts

    def to_feature_dict(self) -> dict[str, float]:
        """Exposes twin_state features for B6 fusion model."""
        return {
            "twin_baseline_glucose": float(self.baseline_glucose),
            "twin_meal_gain": float(self.meal_response_gain),
            "twin_param_uncertainty": float(self.param_uncertainty),
            "twin_completeness_pct": float(self.data_completeness_pct)
        }

# Decision Log

### Decision 001: Target Definition and Decision Interval
- **Condition:** Prediabetes and Type 2 Diabetes (T2D), India Context.
- **Target:** Glucose excursion above 180 mg/dL within next 120 minutes for patients currently below 180 mg/dL.
- **Decision Interval:** Evaluated every 15 minutes.
- **Rationale:** Aligns with standard clinical threshold for postprandial hyperglycemia.

### Decision 002: Dual-CGM Selection in Datasets
- **Choice:** For datasets containing dual CGMs (e.g. Blinded vs Unblinded or primary vs backup), select primary continuous monitoring sensor with highest temporal completeness.
- **Rationale:** Ensures clean, contiguous 5-minute sampling grid without missingness introduced by switching sensors.

### Decision 003: Architectural Separation of Counterfactuals
- **Choice:** Counterfactual predictions combine ML baseline forecaster + mechanistic delta derived from fitted `PatientState` parameters and intervention specifications in `configs/interventions.yaml`.
- **Rationale:** Avoids over-promising purely ML-based causal inference on observational data while providing actionable what-if guidance.

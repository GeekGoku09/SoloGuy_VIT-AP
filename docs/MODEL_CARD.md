# Model Card: Multi-Stream Gradient Boosted Fusion Forecaster (Digital Twin)

## Model Details
- **Model Name:** LightGBM Fusion Forecaster (B5 / B6)
- **Model Architecture:** Gradient Boosted Decision Trees with Isotonic Probability Calibration & Quantile Regression Bands
- **Version:** 0.1.0 (PoC Sandbox)
- **Date:** October 2026
- **Developers:** TEAM_NAME

## Intended Use
- **Primary Use Case:** Real-time 120-minute forward prediction of postprandial glucose excursions (> 180 mg/dL) for patients with prediabetes and Type 2 Diabetes in an Indian healthcare context.
- **Target Audience:** Clinicians, endocrinologists, and digital health researchers.
- **Out of Scope:** Automated clinical decision making, autonomous insulin dosing, direct emergency medical alert system.

## Feature Groups
1. **CGM:** Current glucose level, 15m/30m/60m slopes, 1h/3h/6h rolling mean, SD, CV.
2. **Meals:** Time since last meal, carbs (g), protein (g), fat (g), 2h cumulative carbs and kcal.
3. **Wearables:** 30m/60m/180m step counts, MET activity intensity, 30m mean heart rate, resting HR deviation, sleep stage, HRV RMSSD.
4. **EHR:** Age, BMI, waist circumference, HbA1c, fasting glucose/insulin, hypertension/dyslipidaemia flags, medication profile, TCF7L2 risk alleles.
5. **Twin State:** Fitted physiological baseline glucose, meal response gain parameter, parameter uncertainty.

## Performance & Metrics
- **Evaluated On:** 5-Fold Patient-Grouped Cross Validation (GroupKFold)
- **PR-AUC:** Reported with 95% patient-bootstrapped CIs in `results/ablation.csv`.
- **ROC-AUC:** Reported with 95% patient-bootstrapped CIs in `results/ablation.csv`.
- **False Alert Rate:** Target < 0.5 false alerts per patient-day with 60-minute alert cooldown.

## Ethical & Governance Considerations
- **Data Privacy:** DPDP Act & HIPAA sandbox compliant. Uses only anonymized open-source or synthetic data.
- **Causal Disclaimer:** What-if scenarios represent mechanistic simulations under stated assumptions, not observational causal estimates.
- **Medical Device Status:** Research Prototype Only. Not a certified medical device.

# Doctor-Facing Dashboard Wireframe & Workflow

## Overview
The Streamlit doctor-facing dashboard displays real-time telemetric streams, 2-hour forward predictions, SHAP explainability drivers, static EHR clinical context, counterfactual what-if simulations, and 14-day Ambulatory Glucose Profile (AGP) metrics.

## Doctor's 5-Step Workflow
1. **Open Patient:** Select patient ID from the dropdown; review header sync status, demographics, HbA1c, and synthetic warnings.
2. **Review Risk:** Drag the replay slider to inspect live 2-hour trajectory prediction relative to the 180 mg/dL clinical threshold and 10-90% quantile bands.
3. **Inspect Drivers:** Examine the plain-English SHAP drivers (e.g. current glucose velocity, recent meal carbs, genetic alleles) alongside fasting labs and medication flags.
4. **Try What-If:** Select counterfactual interventions (15-min post-meal walk, meal swap to low GI roti+dal, shifting dinner time) to evaluate simulated peak glucose and risk reductions with clear provenance badges.
5. **Record Plan:** Document personalized glycemic goals and lifestyle interventions based on model predictions and AGP statistics (TIR, TAR, TBR, CV).

## Panel Structure
- **Panel 1:** Patient Header (Twin sync status, completeness, demographics)
- **Panel 2:** Replay Slider & 2-Hour Trajectory Forecast Plot (180 mg/dL line, 10-90% bands)
- **Panel 3:** Excursion Risk Card & Plain-English Drivers
- **Panel 4:** EHR & Clinical Profile Card
- **Panel 5:** What-If Counterfactual Simulator
- **Panel 6:** 14-Day Ambulatory Glucose Profile (AGP: TIR 70-180, TAR >180, TBR <70, CV%)

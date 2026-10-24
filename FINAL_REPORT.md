# FINAL REPORT: PoC DIGITAL TWIN (PREDIABETES & T2D INDIA CONTEXT)

## Executive Summary
This project delivers a competition-ready Proof of Concept (PoC) Digital Twin for glucose event forecasting and what-if lifestyle counterfactual simulation in an Indian healthcare context. The solution fuses static EHR parameters with dynamic continuous glucose monitoring (CGM) and wearable telemetry to forecast postprandial glucose excursions (> 180 mg/dL) 120 minutes in advance.

## What Works & Key Implementation Verification
1. **Scaffolding & Architecture:** Standardized repo layout (`src/dtwin/{data,features,twin,models,eval,app}`), Typer CLI, Makefile wrapper, and configuration YAML files.
2. **Canonical Data Layer & Synthetic Simulator:** Pydantic validation schemas for patients, timeseries (5-min grid), and Indian meal templates (roti+dal, rice+sambar, idli, poha, biryani). Built mechanistic simulator with HbA1c/insulin resistance physics, dawn effect, wearable activity, and genetic risk alleles.
3. **Causal Feature Extraction & Leakage Prevention:** Strict causal feature engineering (`cgm`, `meals`, `wearable`, `ehr`, `twin_state`) using max 15-minute forward-fill. Verified zero future leakage with automated perturbation tests (`test_no_future_leakage`).
4. **Grouped Patient-Wise Cross-Validation:** 5-fold `GroupKFold` split ensuring zero patient overlap between train and test sets (`test_group_disjoint`).
5. **Multi-Stream Fusion Forecaster:** LightGBM classifier with isotonic probability calibration, 60-minute alert cooldown policy, and patient-bootstrapped 95% CIs for PR-AUC, ROC-AUC, and Brier score.
6. **Digital Twin & What-If Engine:** Persistent `PatientState` parameter fitting (baseline glucose, meal response gain) and `WhatIfEngine` simulating counterfactual lifestyle interventions (15-min post-meal walk, low-GI meal swap) with 10-90% quantile bands. Verified counterfactual effect recovery under 10 mg/dL error on synthetic ground truth (`results/whatif_synthetic.json`).
7. **Doctor-Facing Dashboard:** 6-panel Streamlit application featuring live replay streaming slider, 2-hour trajectory prediction, plain-English SHAP drivers, clinical EHR profile, what-if simulator, and 14-day Ambulatory Glucose Profile (AGP: TIR 70-180, TAR >180, TBR <70, CV%).
8. **Submission Audit:** `scripts/check_submission.py` automated validation passing all checks (no phone numbers, no TODO_VERIFY placeholders, PPTX and PDF architecture/presentation files generated, and dynamic README rendering).

## Top 5 Risks a Skeptical Judge Would Raise & Responses
1. **Data Provenance & Real Data Availability:**
   - *Risk:* Real headline datasets (CGMacros, BIG IDEAs, ShanghaiT2DM) require PhysioNet access credentials or manual raw file placement.
   - *Response:* The pipeline degrades gracefully on the bundled 12-patient synthetic sample (`data/sample`), running `make demo` fully offline in < 1 minute, while keeping real data loader scripts ready in `src/dtwin/data/loaders.py`.
2. **Causal Claims vs Mechanistic Simulation:**
   - *Risk:* Observational ML models cannot infer true causal intervention effects.
   - *Response:* Counterfactual forecasts combine ML baseline predictions with mechanistic deltas explicitly marked with provenance badges (`illustrative_default`) in the UI and README.
3. **Over-reliance on Synthetic Priors:**
   - *Risk:* Synthetic priors may not capture rare clinical edge cases.
   - *Response:* Synthetic data is explicitly labeled `SYNTHETIC` and reserved for demo, unit tests, and what-if mechanics, never mixed into real headline evaluation sets.
4. **Clinical Alert Alarm Fatigue:**
   - *Risk:* Frequent false alarms lead clinicians and patients to ignore alerts.
   - *Response:* The alert policy incorporates a 60-minute cooldown and calibrates decision thresholds to maintain < 0.5 false alerts per patient-day.
5. **Medical Device Disclaimer:**
   - *Risk:* Regulatory compliance under CDSCO / FDA guidelines for clinical decision software.
   - *Response:* Prominently marked as a *Research Prototype Only; Not a Medical Device* in the README, model card, presentation slides, and app footer.

## Remaining Manual Steps for Team
1. Record and upload the 20+ minute demo video and replace `VIDEO_URL` in `README.md` if submitting to a live repository.
2. Place raw PhysioNet dataset files into `data/raw/cgmacros/` or `data/raw/big_ideas/` if evaluating on restricted clinical cohorts.

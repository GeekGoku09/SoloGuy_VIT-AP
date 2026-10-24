# Demo Video Presentation Script (~22 Minutes)

## Overview
- **Speaker:** Lead ML / Health-Data Engineer
- **Pacing:** ~140 spoken words per minute (~3,000 words total)
- **Target Audience:** Expert Technical & Clinical Evaluation Panel

---

### Section 1: Problem & India Context (00:00 - 03:00)
- **Slide 1:** Title & Team Overview.
- **Slide 2:** The Prediabetes & T2D Crisis in India (101 million diabetes, 136 million prediabetes per ICMR-INDIAB 2023).
- **Slide 3:** Acute Postprandial Excursions (> 180 mg/dL) as key driver of microvascular and macrovascular complications.

### Section 2: Why a Digital Twin Architecture? (03:00 - 06:00)
- **Slide 4:** Multi-stream telemetric fusion (static EHR + dynamic CGM & wearables).
- **Slide 5:** Persistent PatientState: Online tracking of physiological baseline and meal response gain.
- **Slide 6:** Privacy & Regulatory Boundary: DPDP Act and HIPAA sandbox compliance.

### Section 3: Causal Feature Pipeline & Label Engineering (06:00 - 09:00)
- **Slide 7:** 120-minute forward horizon label y(t) evaluated on 15-minute decision grid.
- **Slide 8:** Strict causal feature building: max 15-minute forward-fill, zero future leakage (proved via automated perturbation tests).

### Section 4: Fusion Forecaster & Model Ablations (09:00 - 13:00)
- **Slide 9:** Hierarchy of models B0 to B6 (LightGBM + Isotonic Calibration).
- **Slide 10:** Grouped 5-Fold Cross Validation protocol and patient-level bootstrap 95% CIs.
- **Slide 11:** Ablation Results: Demonstration of PR-AUC gains as meal, wearable, EHR, and twin_state streams are fused.

### Section 5: Doctor Dashboard & What-If Live Demo (13:00 - 18:00)
- **Slide 12:** Panel 1 & 2: Replay Slider & 2-Hour Trajectory Forecast with 10-90% Quantile Bands.
- **Slide 13:** Panel 3 & 4: Risk Card, SHAP Explainability Drivers in Plain English, Fasting Labs, Medications.
- **Slide 14:** Panel 5: What-If Counterfactual Simulator (15-min post-meal walk, meal swap to low-GI roti+dal).
- **Slide 15:** Panel 6: 14-Day Ambulatory Glucose Profile (AGP) - TIR 70-180, TAR >180, TBR <70, CV%.

### Section 6: Ethics, Limitations & Roadmap (18:00 - 22:00)
- **Slide 16:** Research prototype disclaimer & no-causal-claim boundaries.
- **Slide 17:** Reproducibility (`python -m dtwin demo`), pinned environment, offline execution.
- **Slide 18:** Conclusion & Q&A.

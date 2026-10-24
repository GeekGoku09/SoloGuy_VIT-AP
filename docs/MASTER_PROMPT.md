# ROLE
You are a senior ML engineer and health-data scientist. Build a competition-ready proof of concept (PoC), end to end, in a new project folder inside the current directory. Work autonomously through the phases below. Do not ask me questions except where a rule below allows it; otherwise pick a sensible default, record it in DECISIONS.md and keep going.

# THE CHALLENGE (treat as the spec)
Build a PoC Digital Twin: a virtual replica of one patient that fuses (1) static EHR data (demographics, diagnoses, labs, genetic markers) with (2) dynamic wearable/IoT time series (CGM, heart rate, steps, sleep, HRV) and predicts an adverse event before it happens. It needs a doctor-facing dashboard. Only anonymized, open-source or synthetic data may be used (DPDP Act and HIPAA sandbox rule). A technical and expert panel will judge it; they reward a real fusion of both streams, honest evaluation, a clear doctor workflow and reproducibility.

# DECISIONS ALREADY MADE (do not reopen)
- Condition: prediabetes and type 2 diabetes (T2D), India context.
- Target: for a patient whose glucose is currently below 180 mg/dL, will glucose rise above 180 mg/dL at any point in the next 120 minutes? Horizon, threshold and decision interval live in configs/default.yaml.
- Twin = persistent per-patient state + fusion forecaster + what-if engine (see ARCHITECTURE).
- Real data for headline metrics: CGMacros (PhysioNet). Optional: the BIG IDEAs glycemic/wearable dataset (PhysioNet) for external validation, and ShanghaiT2DM for a T2D-only check. Synthetic data (own generator with Indian-profile priors; Synthea optional) is for the demo, tests and stress tests only.
- Stack: Python 3.11, pandas, pyarrow, scikit-learn, LightGBM, SHAP, PyTorch (optional extra), Streamlit, Plotly, pydantic, typer, pytest, ruff, python-pptx.

# HARD RULES
1. No fabricated results. Every number in the README, slides and dashboard is produced by code and read from results/*.json at build time. Anything not run is written as 'not run'. Never invent citations, clinical effect sizes or dataset statistics; mark unknowns TODO_VERIFY, and make check-submission fail while any remain.
2. Headline metrics use real open data only, with patient-wise splits (leave-one-patient-out, or grouped 5-fold when there are more than 60 patients). Synthetic data is for the demo, unit tests, stress tests and what-if mechanics. Label every synthetic output SYNTHETIC and never mix synthetic rows into a headline train or test set.
3. No leakage. Features at time t use only data stamped <= t, and short gaps are filled by causal forward-fill (max 15 minutes), never by interpolation that looks ahead. Scalers, imputers, thresholds and calibrators are fit on training patients only. Prove it with tests (perturb the future, assert identical features).
4. Never commit raw datasets. Ship download instructions and docs/DATA_PROVENANCE.md (source URL, licence copied from the dataset's own page, access level, de-identification status, how it is used, citation). If you cannot verify a licence offline, mark TODO_VERIFY. Never bypass restricted or credentialed access.
5. What-if results are simulations under stated assumptions, not causal estimates. Every intervention effect sits in configs/interventions.yaml with a provenance field (illustrative_default | fitted_from_data | literature_TODO_VERIFY), and the UI shows that label.
6. Research prototype, not a medical device. Say so in the README, model card and app footer. No diagnosis or treatment instructions anywhere.
7. Privacy: the repo contains no phone numbers or personal emails. The team leader's contact details go only on the submission form.
8. Reproducible: pinned dependencies, fixed seeds, pathlib paths. All commands are python -m dtwin <command>; in this prompt 'make X' means python -m dtwin X, and the Makefile only wraps the CLI so Windows works too. make demo must run offline on the bundled synthetic sample in under 5 minutes on a laptop CPU.
9. Missing dataset? Degrade gracefully: skip that module, say so in the report, keep going.

# ARCHITECTURE (build this)
EHR stream (static) ----------+
                              +--> harmonise (canonical schema, 5-minute grid, QC) --> twin core --> doctor dashboard
CGM, wearables, meals --------+

Canonical schema (pydantic validation, parquet storage):
- patients: patient_id, source, is_synthetic, age, sex, bmi, waist_cm, family_history_t2d, diagnosis_group (normal/prediabetes/t2d), past diagnoses and comorbidities (for example hypertension, dyslipidaemia), years_since_diagnosis, hba1c, fasting_glucose, fasting_insulin, homa_ir, lipids, bp, medication flags, tcf7l2_risk_alleles (0-2). Missing = NaN plus availability flags.
- timeseries (5-minute grid, timezone documented): patient_id, ts, glucose_mgdl, hr_bpm, steps, met, sleep_stage, hrv_rmssd.
- meals: patient_id, ts, carbs_g, protein_g, fat_g, fiber_g, kcal, label.
Feature groups: cgm, meals, wearable, ehr, twin_state. Each dataset declares which groups it provides (real data may lack sleep, HRV or genetic markers); ablations use only the groups that exist for that dataset.

Twin core:
- PatientState: EHR embedding plus physiological parameters fitted per patient (meal-response gain, delay and decay, baseline glucose, insulin-sensitivity proxy, resting heart-rate baseline). It updates online with every new reading (recursive or exponentially weighted least squares, or a Kalman filter) and reports sync status: last update, data completeness, parameter uncertainty.
- FusionForecaster: (a) calibrated probability of glucose > 180 mg/dL within 120 minutes; (b) estimated time to crossing; (c) glucose trajectory at 15, 30, 60, 90 and 120 minutes with a 10-90% band (quantile regression, optionally conformal-adjusted on training patients); (d) SHAP top drivers rendered as plain-English sentences.
- WhatIfEngine: re-runs the forecast for a scenario (swap the next meal for a preset Indian meal from configs/meals_india.yaml, add a post-meal walk, shift dinner time). Counterfactual = ML baseline forecast + a mechanistic delta computed from the patient's fitted parameters and the intervention effect, with a wider band for assumption uncertainty. Show the change in predicted peak and in risk.

# REPO LAYOUT
src/dtwin/{data,features,twin,models,eval,app}, configs/, scripts/, tests/, data/{sample,raw,processed}, results/figures, docs/screenshots, notebooks/ (optional).
CLI: python -m dtwin <setup|data-sample|inspect|features|train|eval|whatif|app|docs|check-submission|demo|test>.

# PHASES (run in order; after each, run its gate, tick PROGRESS.md, commit 'phase N: ...', then continue automatically)

PHASE 0 - Scaffold.
Create the project folder, git init, .gitignore (data/raw, data/processed, .venv), pyproject.toml with pinned versions, the CLI and Makefile wrapper, configs/default.yaml (thresholds, horizons, seeds, paths), configs/project.yaml (the INPUTS above), LICENSE (CODE_LICENSE, year 2026, holder TEAM_NAME), PROGRESS.md (every phase as a checkbox), DECISIONS.md, a docs/DATA_PROVENANCE.md skeleton, a README skeleton, and a verbatim copy of this prompt at docs/MASTER_PROMPT.md.
Gate: python -m dtwin --help works and pytest runs.

PHASE 1 - Data layer.
a) Canonical schema and validators.
b) Synthetic generator (src/dtwin/data/synthetic): default 120 patients, 14 days each, 5-minute grid; bundle a 12-patient sample under data/sample (under 10 MB). Simulate mechanistically: baseline glucose from HbA1c via estimated average glucose (eAG = 28.7 x HbA1c - 46.7); a meal-response kernel (delay, rise, decay) whose gain scales with insulin resistance and meal carbs; a dawn effect; a lower response after post-meal activity; poor previous-night sleep raising next-day insulin resistance (state it as an assumption); CGM noise, drift and dropouts; heart rate from activity and circadian rhythm; sleep stages and HRV; Indian meal templates (rice, roti, idli, dosa, dal, poha, upma, biryani, sweets); Indian-profile priors in configs/synthetic_priors.yaml labelled source: illustrative; genetic risk as a risk-allele count. The generator must also simulate the same patient with and without an intervention, to give ground-truth counterfactuals for Phase 4.
c) Real loaders for CGMacros, BIG IDEAs and ShanghaiT2DM. FIRST write scripts/inspect_dataset.py (files, columns, dtypes, units, time range, sampling interval, missingness) and save docs/data_dictionary_observed.md. Never hard-code column names from memory. Convert units explicitly (mmol/L x 18.016 = mg/dL). If a dataset has two CGMs, pick one and justify it in DECISIONS.md. Files are expected in data/raw/<dataset>/; if absent, print exact manual download steps.
d) Data audit: per-patient coverage, gaps, glucose distribution, and event prevalence at 140, 160 and 180 mg/dL, saved to results/data_audit.json plus a figure. The primary target stays 180; if positives are too sparse for stable evaluation, say so in the report instead of changing the target after seeing results.
e) Optional: scripts/synthea_import.py mapping Synthea CSV output to the canonical EHR table (skip if Java or Synthea is absent).
Gate: schema tests pass, data-sample builds in under 2 minutes, and the audit runs on the sample (and on real data if present).

PHASE 2 - Labels, features, splits.
Label y(t) = 1 if the maximum glucose in (t, t + 120 min] exceeds 180, evaluated only at times where glucose(t) < 180 and at least 80% of CGM readings in the window are present. Decision times: every 15 minutes. Features: glucose level and slopes (15, 30, 60 min); rolling mean, SD and CV over 1, 3 and 6 h; time since last meal; last-meal and 2-hour carbs, protein, fat, fibre and kcal; steps and METs over 30, 60 and 180 min; heart-rate mean and deviation from the personal resting baseline; sleep duration, efficiency and HRV where available; time-of-day sin/cos; EHR features. (twin_state features arrive in Phase 4.) Splits: grouped by patient only, with an inner group split for tuning and calibration. Also run a sensitivity variant that drops samples within 60 minutes of a previous excursion.
Gate: test_no_future_leakage, test_group_disjoint and test_label_definition (on a hand-built series) pass.

PHASE 3 - Baselines, fusion model, evaluation (completes the first end-to-end slice).
Models behind one interface: B0 trend extrapolation of the last 30 minutes; B1 LightGBM on cgm only; B2 + meals; B3 + wearable; B4 + EHR; B5 full fusion (cgm, meals, wearable and ehr groups). Calibrate probabilities with patient-grouped isotonic or Platt scaling. Alert policy: choose the threshold on training patients to meet a target false-alert rate, with a 60-minute cooldown after each alert. Metrics, each with a 95% CI from a patient-level bootstrap of out-of-fold predictions: PR-AUC against the prevalence line, ROC-AUC, Brier score and calibration curve, sensitivity at the fixed false-alert rate, and event-level detection. An excursion event is the first reading above 180 after at least 60 minutes below it; it counts as detected if an alert fired 15 to 120 minutes earlier; a false alert is an alert with no event in the next 120 minutes. Report false alerts per patient-day and median lead time. Break results down per cohort (healthy, prediabetes, T2D) and per patient, showing positives per patient. Add a SHAP summary. Write results/ablation.csv, results/metrics.json and figures under results/figures. Choose the shipped model on inner validation only and report held-out folds once.
Gate: python -m dtwin eval writes all of the above; make demo builds features on the bundled sample, trains B5, prints SYNTHETIC smoke metrics and launches the dashboard (a stub is fine until Phase 6; add a --no-launch flag for tests).

PHASE 4 - Twin layer.
Implement PatientState with online updates, the trajectory head with its 10-90% band, plain-English SHAP explanations and WhatIfEngine. Add the twin_state feature group (parameter estimates from past data only) as B6 = B5 + twin_state, include it in the ablation table and re-select the shipped model on inner validation. Plot parameter convergence over a patient's first days. Meal presets in configs/meals_india.yaml are approximate; cross-check them against the Indian Food Composition Tables (IFCT 2017) when you can, otherwise mark TODO_VERIFY. Tests: a zero-change scenario equals the baseline; more carbs raises the predicted peak; a post-meal walk lowers it; the band widens under what-if. On SYNTHETIC data only, compare the engine's predicted intervention deltas with the generator's ground-truth counterfactuals and report the error as 'recovery of simulated effects (synthetic)'.
Gate: twin tests pass and results/whatif_synthetic.json exists.

PHASE 5 - Optional extras (only if the Phase 0-4 gates are green).
A GRU or temporal CNN with static features as conditioning, on the same splits as the best LightGBM model; keep whichever wins on inner validation and report the other too. If the neural model does not beat LightGBM, say so plainly. External check: train on CGMacros, test on BIG IDEAs without retuning, using only the feature groups both share, and report the drop as domain shift. Add a synthetic Indian-profile stress test, labelled directional only.

PHASE 6 - Doctor dashboard (Streamlit).
Six panels: (1) patient header with twin sync status, data completeness and model version; (2) glucose with the 2-hour forecast band and the 180 line, driven by a replay slider that simulates live streaming from recorded data and is labelled as replay; (3) risk card with probability, estimated time to crossing and top drivers in plain English; (4) EHR card with past diagnoses, labs, medications and genetic risk; (5) what-if panel (meal swap, post-meal walk, dinner shift) showing the counterfactual curve, the change in peak and risk, the band and the provenance badge; (6) 14-day ambulatory glucose profile with time in range 70-180, time above 180, time below 70, mean glucose, GMI and CV. Add a SYNTHETIC banner for synthetic patients, a research-prototype footer, colourblind-safe colours and no real names. Load precomputed artifacts so it starts in under 10 seconds. Write docs/DASHBOARD_WIREFRAME.md with the doctor's five-step workflow (open patient, review risk, inspect drivers, try what-if, record plan). Save screenshots of all six panels to docs/screenshots (use Playwright if available; otherwise list the exact shots for me to take).
Gate: the app boots and a streamlit.testing.v1.AppTest smoke test passes.

PHASE 7 - Submission Artifacts.
- README.md with these items in this order: project title (H1); team and college/incubator details (names and roles only); problem statement and healthcare use case (India context; cite only figures listed in docs/SOURCES.md, which you seed with the ICMR-INDIAB 2023 estimates of about 101 million people with diabetes and 136 million with prediabetes, marked TODO_VERIFY until you have opened the paper); technical stack and AI/ML model details; demo video link (VIDEO_URL, at least 20 minutes); open-source licence details (code licence, dataset licences from DATA_PROVENANCE, third-party licences via pip-licenses); architecture diagram link (PDF/PPT); presentation link (PDF/PPT). Then Results (rendered from results/*.json by scripts/render_results.py), Quick start and Limitations.
- docs/architecture.pptx (editable shapes) and docs/architecture.pdf, generated by script: data sources, harmonisation, twin core, model heads, dashboard, and the privacy boundary (only anonymized or synthetic data).
- docs/presentation.pptx and docs/presentation.pdf, about 20 slides with numbers pulled from results/: problem and India context, why a twin, data streams, architecture, twin core, labels and evaluation protocol, results, fusion ablation, calibration, event-level detection, explainability, what-if demo, dashboard walkthrough, limitations and ethics (DPDP/HIPAA, synthetic data, domain shift, no causal claims), roadmap, licence, team. Export PDFs with LibreOffice headless if installed, otherwise build the PDF directly.
- docs/video_script.md: a timed script for a demo video of at least 20 minutes (aim for 22; about 140 spoken words per minute), slide by slide, with the exact demo clicks. You cannot record the video; I will.
- docs/MODEL_CARD.md and a completed docs/DATA_PROVENANCE.md.
- check-submission (scripts/check_submission.py) scans README, configs and docs (excluding docs/MASTER_PROMPT.md) and fails on: a folder name that does not match configs/project.yaml; a missing required README item or LICENSE; any placeholder or TODO_VERIFY left; a missing architecture or presentation file in PDF or PPT; VIDEO_URL not a URL; raw data or any file over 50 MB in git; phone-number-like strings; a README Results section that differs from a fresh render of results/*.json.
Gate: check-submission runs, and the only failures left are items only the team can supply (video link, team details, licence checks).

PHASE 8 - Hardening.
Clone the repo into a temp folder, create a fresh virtual environment, run make demo and make test, fix whatever breaks, run ruff, and make sure every README command is copy-paste correct. Finish with FINAL_REPORT.md: what works, what was verified, what is not done, the top five risks a skeptical judge would raise, and the exact steps I still have to do by hand.

# WORKING AGREEMENTS
- Begin by writing PROGRESS.md and a one-page plan, then execute. After each phase: run its gate, tick PROGRESS.md, commit, and continue automatically.
- Vertical slice first: by the end of Phase 3 a stranger can run make demo. Depth comes after.
- If a gate fails, fix the root cause; do not weaken the test. If blocked by missing network or data, record it in DECISIONS.md, use the degrade path and continue.
- Small typed modules (under about 300 lines each), docstrings that state units (mg/dL, minutes) and timezone conventions. Notebooks are never the source of truth.
- You may ask me only for dataset files or credentials, and before any change to the target definition. Everything else: decide, log, continue.
- Do not push to any remote, and do not modify or delete anything outside the project folder (a temp folder for the Phase 8 fresh-clone test is fine); I will publish the repo myself.
- Do not claim anything is done unless its gate passed. Your final message is a short summary plus the Phase 8 report.

Start now with Phase 0.

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_priors(priors_path: str = "configs/synthetic_priors.yaml") -> dict[str, Any]:
    with open(priors_path, "r") as f:
        return yaml.safe_load(f)

def load_meals_config(meals_path: str = "configs/meals_india.yaml") -> dict[str, Any]:
    with open(meals_path, "r") as f:
        return yaml.safe_load(f)["meals"]

def generate_synthetic_cohort(
    num_patients: int = 120,
    days: int = 14,
    seed: int = 42,
    priors_path: str = "configs/synthetic_priors.yaml",
    meals_path: str = "configs/meals_india.yaml"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic cohort conforming to canonical schema.
    Returns (patients_df, timeseries_df, meals_df).
    """
    rng = np.random.default_rng(seed)
    priors = load_priors(priors_path)
    meal_templates = load_meals_config(meals_path)
    meal_keys = list(meal_templates.keys())

    patients_list = []
    ts_list = []
    meals_list = []

    start_date = datetime(2026, 1, 1, 0, 0, 0)
    total_steps = days * 24 * 12  # 5-min intervals

    for i in range(num_patients):
        pid = f"SYNTH_{i+1:03d}"
        
        # Diagnosis group assignment
        diag_prob = rng.multinomial(1, [priors["demographics"]["normal_prev"],
                                        priors["demographics"]["prediabetes_prev"],
                                        priors["demographics"]["t2d_prev"]])
        diag_idx = np.argmax(diag_prob)
        diag_group = ["normal", "prediabetes", "t2d"][diag_idx]

        age = float(np.clip(rng.normal(priors["demographics"]["age_mean"], priors["demographics"]["age_std"]), 20, 80))
        bmi = float(np.clip(rng.normal(priors["demographics"]["bmi_mean"], priors["demographics"]["bmi_std"]), 18.0, 42.0))
        waist_cm = bmi * 3.2 + rng.normal(0, 4)
        fam_hist = bool(rng.random() < priors["demographics"]["family_history_prob"])
        tcf7l2 = int(rng.binomial(2, priors["demographics"]["tcf7l2_risk_allele_freq"]))

        if diag_group == "normal":
            hba1c = rng.uniform(*priors["physiology"]["hba1c_normal"])
            years_dx = 0.0
            metformin = False
        elif diag_group == "prediabetes":
            hba1c = rng.uniform(*priors["physiology"]["hba1c_prediabetes"])
            years_dx = rng.uniform(0.5, 4.0)
            metformin = bool(rng.random() < 0.2)
        else:
            hba1c = rng.uniform(*priors["physiology"]["hba1c_t2d"])
            years_dx = rng.uniform(1.0, 15.0)
            metformin = True

        # Mechanistic baseline glucose from HbA1c (eAG = 28.7 * HbA1c - 46.7)
        eAG = 28.7 * hba1c - 46.7
        baseline_glucose = float(np.clip(eAG, 80.0, 220.0))
        
        # Insulin resistance factor (scales response to carbs)
        ir_factor = 1.0 + (hba1c - 5.0) * 0.35 + (0.1 if tcf7l2 > 0 else 0.0)

        patients_list.append({
            "patient_id": pid,
            "source": "synthetic_generator",
            "is_synthetic": True,
            "age": round(age, 1),
            "sex": rng.choice(["M", "F"]),
            "bmi": round(bmi, 1),
            "waist_cm": round(waist_cm, 1),
            "family_history_t2d": fam_hist,
            "diagnosis_group": diag_group,
            "hypertension": bool(rng.random() < 0.35 if diag_group != "normal" else rng.random() < 0.1),
            "dyslipidaemia": bool(rng.random() < 0.40 if diag_group != "normal" else rng.random() < 0.15),
            "years_since_diagnosis": round(years_dx, 1),
            "hba1c": round(hba1c, 2),
            "fasting_glucose": round(baseline_glucose * 0.9, 1),
            "fasting_insulin": round(8.0 * ir_factor, 1),
            "homa_ir": round((baseline_glucose * 0.9 * 8.0 * ir_factor) / 405.0, 2),
            "bp_systolic": 120 + (15 if diag_group == "t2d" else 0),
            "bp_diastolic": 80 + (8 if diag_group == "t2d" else 0),
            "medication_metformin": metformin,
            "medication_sulfonylurea": bool(diag_group == "t2d" and rng.random() < 0.3),
            "medication_dpp4i": bool(diag_group == "t2d" and rng.random() < 0.2),
            "medication_sglt2i": bool(diag_group == "t2d" and rng.random() < 0.15),
            "medication_insulin": bool(diag_group == "t2d" and hba1c > 9.0),
            "tcf7l2_risk_alleles": tcf7l2
        })

        # Generate daily schedule & meals
        time_grid = [start_date + timedelta(minutes=5 * step) for step in range(total_steps)]
        
        # Patient meal schedule (breakfast ~8am, lunch ~1pm, dinner ~8pm)
        patient_meals = []
        for day_idx in range(days):
            day_start = start_date + timedelta(days=day_idx)
            # Breakfast
            t_b = day_start + timedelta(hours=8, minutes=int(rng.normal(0, 15)))
            m_b = rng.choice(meal_keys)
            patient_meals.append((t_b, meal_templates[m_b]))
            # Lunch
            t_l = day_start + timedelta(hours=13, minutes=int(rng.normal(0, 20)))
            m_l = rng.choice(meal_keys)
            patient_meals.append((t_l, meal_templates[m_l]))
            # Dinner
            t_d = day_start + timedelta(hours=20, minutes=int(rng.normal(0, 20)))
            m_d = rng.choice(meal_keys)
            patient_meals.append((t_d, meal_templates[m_d]))

        for t_meal, m_info in patient_meals:
            meals_list.append({
                "patient_id": pid,
                "ts": t_meal,
                "carbs_g": m_info["carbs_g"],
                "protein_g": m_info["protein_g"],
                "fat_g": m_info["fat_g"],
                "fiber_g": m_info["fiber_g"],
                "kcal": m_info["kcal"],
                "label": m_info["name"]
            })

        # Time series signal simulation
        glucose_signal = np.full(total_steps, baseline_glucose)
        hr_signal = np.full(total_steps, 70.0)
        steps_signal = np.zeros(total_steps)
        met_signal = np.ones(total_steps)
        sleep_stage = ["awake"] * total_steps
        hrv_signal = np.full(total_steps, 45.0)

        # Apply circadian dawn effect (glucose rise between 4am - 7am)
        for step, t in enumerate(time_grid):
            hour = t.hour + t.minute / 60.0
            if 4.0 <= hour <= 7.0:
                dawn_delta = np.sin((hour - 4.0) / 3.0 * np.pi) * 15.0 * (ir_factor - 0.5)
                glucose_signal[step] += dawn_delta

            # Sleep stages (11pm to 7am)
            if hour >= 23 or hour < 7:
                sleep_stage[step] = "deep" if (hour >= 1 and hour < 4) else "light"
                hr_signal[step] = 58.0 + rng.normal(0, 2)
                hrv_signal[step] = 55.0 + rng.normal(0, 3)
                met_signal[step] = 0.9
            else:
                hr_signal[step] = 72.0 + rng.normal(0, 4)
                hrv_signal[step] = 40.0 + rng.normal(0, 4)

        # Apply meal response kernel
        for t_meal, m_info in patient_meals:
            carbs = m_info["carbs_g"]
            meal_step = int((t_meal - start_date).total_seconds() / 300)
            
            # Response kernel duration: 180 min (36 steps)
            for k in range(36):
                if meal_step + k < total_steps:
                    t_mins = k * 5
                    # Gamma-like kernel: t * exp(-t / tau)
                    kernel_val = (t_mins / 35.0) * np.exp(-t_mins / 35.0) * 2.718
                    glucose_delta = carbs * 0.8 * ir_factor * kernel_val
                    glucose_signal[meal_step + k] += glucose_delta

            # Post-meal activity (walk after lunch occasionally)
            if t_meal.hour == 13 and rng.random() < 0.5:
                walk_start = meal_step + 6 # 30 mins after lunch
                for w in range(3): # 15 min walk
                    if walk_start + w < total_steps:
                        steps_signal[walk_start + w] = 450
                        met_signal[walk_start + w] = 3.5
                        hr_signal[walk_start + w] = 105.0
                        # Activity lowers glucose excursion
                        glucose_signal[walk_start + w: walk_start + w + 12] *= 0.92

        # Add noise, dropouts, and drift
        noise = rng.normal(0, 3.0, total_steps)
        glucose_signal = np.clip(glucose_signal + noise, 50.0, 400.0)

        # Missingness (2% random dropouts)
        for step in range(total_steps):
            if rng.random() < 0.02:
                g_val = None
            else:
                g_val = float(glucose_signal[step])

            ts_list.append({
                "patient_id": pid,
                "ts": time_grid[step],
                "glucose_mgdl": g_val,
                "hr_bpm": float(hr_signal[step]),
                "steps": float(steps_signal[step]),
                "met": float(met_signal[step]),
                "sleep_stage": sleep_stage[step],
                "hrv_rmssd": float(hrv_signal[step])
            })

    patients_df = pd.DataFrame(patients_list)
    timeseries_df = pd.DataFrame(ts_list)
    meals_df = pd.DataFrame(meals_list)

    return patients_df, timeseries_df, meals_df

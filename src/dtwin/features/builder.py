
import numpy as np
import pandas as pd


def extract_causal_features_at_t(
    t: pd.Timestamp,
    patient_id: str,
    ts_history: pd.DataFrame,
    meals_history: pd.DataFrame,
    patient_static: dict,
    max_ffill_minutes: int = 15
) -> dict[str, float]:
    """
    Extracts causal features at timestamp t using ONLY data stamped <= t.
    Short gaps in continuous signals are causal forward-filled up to max_ffill_minutes.
    """
    features = {}

    # --- CGM GROUP ---
    cgm_hist = ts_history[ts_history["ts"] <= t].sort_values("ts")
    if not cgm_hist.empty and "glucose_mgdl" in cgm_hist.columns:
        # Causal forward fill max 15 minutes
        latest_cgm_ts = cgm_hist["ts"].iloc[-1]
        time_diff_min = (t - latest_cgm_ts).total_seconds() / 60.0
        
        latest_g = cgm_hist["glucose_mgdl"].dropna()
        if not latest_g.empty and time_diff_min <= max_ffill_minutes:
            curr_g = latest_g.iloc[-1]
        else:
            curr_g = np.nan

        features["cgm_glucose_mgdl"] = float(curr_g)

        # Slopes over 15, 30, 60 min
        for slope_m in [15, 30, 60]:
            t_past = t - pd.Timedelta(minutes=slope_m)
            past_sub = cgm_hist[cgm_hist["ts"] >= t_past]["glucose_mgdl"].dropna()
            if len(past_sub) >= 2:
                features[f"cgm_slope_{slope_m}m"] = float((past_sub.iloc[-1] - past_sub.iloc[0]) / slope_m)
            else:
                features[f"cgm_slope_{slope_m}m"] = 0.0

        # Rolling statistics (1h, 3h, 6h)
        for roll_h in [1, 3, 6]:
            t_start = t - pd.Timedelta(hours=roll_h)
            window_vals = cgm_hist[cgm_hist["ts"] >= t_start]["glucose_mgdl"].dropna()
            if len(window_vals) > 0:
                m_val = float(window_vals.mean())
                s_val = float(window_vals.std()) if len(window_vals) > 1 else 0.0
                features[f"cgm_mean_{roll_h}h"] = m_val
                features[f"cgm_std_{roll_h}h"] = s_val
                features[f"cgm_cv_{roll_h}h"] = float((s_val / m_val) * 100.0) if m_val > 0 else 0.0
            else:
                features[f"cgm_mean_{roll_h}h"] = np.nan
                features[f"cgm_std_{roll_h}h"] = np.nan
                features[f"cgm_cv_{roll_h}h"] = np.nan
    else:
        features["cgm_glucose_mgdl"] = np.nan

    # --- MEALS GROUP ---
    m_hist = meals_history[meals_history["ts"] <= t].sort_values("ts")
    if not m_hist.empty:
        last_meal_ts = m_hist["ts"].iloc[-1]
        features["meal_time_since_last_min"] = float((t - last_meal_ts).total_seconds() / 60.0)
        
        last_meal = m_hist.iloc[-1]
        features["meal_last_carbs_g"] = float(last_meal.get("carbs_g", 0.0))
        features["meal_last_protein_g"] = float(last_meal.get("protein_g", 0.0))
        features["meal_last_fat_g"] = float(last_meal.get("fat_g", 0.0))

        # 2-hour window meal sums
        t_2h = t - pd.Timedelta(hours=2)
        m_2h = m_hist[m_hist["ts"] >= t_2h]
        features["meal_2h_carbs_g"] = float(m_2h["carbs_g"].sum()) if not m_2h.empty else 0.0
        features["meal_2h_kcal"] = float(m_2h["kcal"].sum()) if not m_2h.empty else 0.0
    else:
        features["meal_time_since_last_min"] = 9999.0
        features["meal_last_carbs_g"] = 0.0
        features["meal_last_protein_g"] = 0.0
        features["meal_last_fat_g"] = 0.0
        features["meal_2h_carbs_g"] = 0.0
        features["meal_2h_kcal"] = 0.0

    # --- WEARABLE GROUP ---
    if not cgm_hist.empty:
        # Steps and METs over 30, 60, 180 min
        for w_m in [30, 60, 180]:
            t_w = t - pd.Timedelta(minutes=w_m)
            w_sub = cgm_hist[cgm_hist["ts"] >= t_w]
            features[f"wearable_steps_{w_m}m"] = float(w_sub["steps"].sum()) if "steps" in w_sub.columns else 0.0
            features[f"wearable_met_mean_{w_m}m"] = float(w_sub["met"].mean()) if "met" in w_sub.columns and len(w_sub) > 0 else 1.0

        # HR mean & deviation from baseline
        if "hr_bpm" in cgm_hist.columns:
            hr_sub = cgm_hist[cgm_hist["ts"] >= (t - pd.Timedelta(minutes=30))]["hr_bpm"].dropna()
            hr_mean = float(hr_sub.mean()) if len(hr_sub) > 0 else 70.0
            features["wearable_hr_mean_30m"] = hr_mean
            resting_hr = patient_static.get("resting_hr", 70.0)
            features["wearable_hr_dev_resting"] = float(hr_mean - resting_hr)

        # Sleep & HRV
        features["wearable_sleep_stage"] = str(cgm_hist["sleep_stage"].iloc[-1]) if "sleep_stage" in cgm_hist.columns and not cgm_hist.empty else "awake"
        features["wearable_hrv_rmssd"] = float(cgm_hist["hrv_rmssd"].dropna().iloc[-1]) if "hrv_rmssd" in cgm_hist.columns and not cgm_hist["hrv_rmssd"].dropna().empty else np.nan

    # Time of day sin/cos
    tod_hours = t.hour + t.minute / 60.0
    features["tod_sin"] = float(np.sin(2 * np.pi * tod_hours / 24.0))
    features["tod_cos"] = float(np.cos(2 * np.pi * tod_hours / 24.0))

    # --- EHR GROUP ---
    features["ehr_age"] = float(patient_static.get("age", 50.0))
    features["ehr_bmi"] = float(patient_static.get("bmi", 25.0))
    features["ehr_hba1c"] = float(patient_static.get("hba1c", 6.0))
    features["ehr_fasting_glucose"] = float(patient_static.get("fasting_glucose", 100.0))
    features["ehr_tcf7l2_risk_alleles"] = int(patient_static.get("tcf7l2_risk_alleles", 0))
    features["ehr_diag_group"] = str(patient_static.get("diagnosis_group", "prediabetes"))

    return features

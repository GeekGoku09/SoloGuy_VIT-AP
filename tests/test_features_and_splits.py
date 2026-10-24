from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from dtwin.features.builder import extract_causal_features_at_t
from dtwin.features.labels import compute_labels_for_patient
from dtwin.features.splits import create_patient_grouped_splits


def test_label_definition():
    start = datetime(2026, 1, 1, 10, 0)
    # Hand-built series: glucose starts at 120, jumps to 190 at minute 45
    times = [start + timedelta(minutes=5*i) for i in range(30)] # 150 mins
    values = [120.0]*8 + [190.0]*10 + [140.0]*12
    ts_df = pd.DataFrame({"patient_id": "P01", "ts": times, "glucose_mgdl": values})

    labels_df = compute_labels_for_patient(ts_df, threshold_mgdl=180.0, horizon_minutes=120, decision_interval_minutes=15)
    
    assert not labels_df.empty
    # At t=0 (10:00), max glucose in next 120m is 190 -> y=1
    first_label = labels_df.iloc[0]
    assert first_label["y"] == 1
    assert first_label["max_glucose_horizon"] == 190.0

def test_no_future_leakage():
    t_decision = datetime(2026, 1, 1, 12, 0)
    start = datetime(2026, 1, 1, 10, 0)
    
    # Original time series up to 12:00
    times = [start + timedelta(minutes=5*i) for i in range(25)] # up to 12:00
    values = [110.0 + i for i in range(25)]
    ts_df_orig = pd.DataFrame({"patient_id": "P01", "ts": times, "glucose_mgdl": values, "steps": [0.0]*25, "met": [1.0]*25, "hr_bpm": [70.0]*25})

    # Perturbed future time series (after 12:00, e.g. massive spike at 12:05)
    future_times = times + [t_decision + timedelta(minutes=5*i) for i in range(1, 10)]
    future_values = values + [500.0]*9
    ts_df_future = pd.DataFrame({"patient_id": "P01", "ts": future_times, "glucose_mgdl": future_values, "steps": [0.0]*34, "met": [1.0]*34, "hr_bpm": [70.0]*34})

    meals_df = pd.DataFrame(columns=["ts", "carbs_g", "protein_g", "fat_g", "kcal"])
    patient_static = {"age": 50, "bmi": 25, "hba1c": 6.0}

    feat_orig = extract_causal_features_at_t(t_decision, "P01", ts_df_orig, meals_df, patient_static)
    feat_future = extract_causal_features_at_t(t_decision, "P01", ts_df_future, meals_df, patient_static)

    # Features at t_decision must be IDENTICAL regardless of future perturbation
    for key in feat_orig:
        val_orig = feat_orig[key]
        val_future = feat_future[key]
        if isinstance(val_orig, float) and np.isnan(val_orig):
            assert np.isnan(val_future)
        else:
            assert val_orig == val_future, f"Leakage detected in feature {key}: {val_orig} != {val_future}"

def test_group_disjoint():
    df = pd.DataFrame({
        "patient_id": [f"P{i:02d}" for i in range(20)] * 5,
        "feature_1": np.random.randn(100),
        "target": np.random.randint(0, 2, 100)
    })

    for train_idx, test_idx in create_patient_grouped_splits(df, n_splits=5):
        train_patients = set(df.iloc[train_idx]["patient_id"])
        test_patients = set(df.iloc[test_idx]["patient_id"])
        assert len(train_patients.intersection(test_patients)) == 0

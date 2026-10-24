import numpy as np
import pandas as pd

from dtwin.eval.metrics import bootstrap_metric_ci, evaluate_alert_policy
from dtwin.models.forecasters import B0TrendForecaster, LightGBMFusionForecaster


def test_models_and_metrics():
    # Synthetic feature table
    X = pd.DataFrame({
        "cgm_glucose_mgdl": [120.0, 150.0, 175.0, 110.0, 160.0],
        "cgm_slope_30m": [0.5, 1.2, 0.1, -0.2, 0.8],
        "meal_last_carbs_g": [30.0, 60.0, 10.0, 0.0, 50.0],
        "wearable_steps_30m": [0.0, 100.0, 500.0, 1000.0, 50.0],
        "ehr_hba1c": [5.5, 6.8, 8.2, 5.2, 7.1]
    })
    y = np.array([0, 1, 1, 0, 1])
    patient_ids = np.array(["P1", "P1", "P2", "P2", "P3"])

    # Test B0
    b0 = B0TrendForecaster()
    probs_b0 = b0.predict_proba(X)
    assert len(probs_b0) == 5
    assert np.all((probs_b0 >= 0.0) & (probs_b0 <= 1.0))

    # Test LightGBM Fusion
    lgb = LightGBMFusionForecaster(feature_groups=["cgm", "meals", "wearable", "ehr"])
    lgb.fit(X, y)
    probs_lgb = lgb.predict_proba(X)
    assert len(probs_lgb) == 5

    # Test bootstrap metrics
    ci_res = bootstrap_metric_ci(y, probs_lgb, patient_ids, n_bootstraps=10)
    assert "pr_auc" in ci_res
    assert "roc_auc" in ci_res

    # Test alert evaluation
    ts = pd.date_range("2026-01-01", periods=5, freq="1h")
    df_eval = pd.DataFrame({"patient_id": patient_ids, "ts": ts, "y": y})
    alerts = evaluate_alert_policy(df_eval, probs_lgb, threshold=0.5)
    assert "false_alerts_per_patient_day" in alerts

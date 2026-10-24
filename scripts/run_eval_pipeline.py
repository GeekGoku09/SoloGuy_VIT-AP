import json
from pathlib import Path

import numpy as np
import pandas as pd

from dtwin.data.synthetic import generate_synthetic_cohort
from dtwin.eval.metrics import bootstrap_metric_ci, evaluate_alert_policy
from dtwin.features.builder import extract_causal_features_at_t
from dtwin.features.labels import compute_labels_for_patient
from dtwin.features.splits import create_patient_grouped_splits
from dtwin.models.forecasters import B0TrendForecaster, LightGBMFusionForecaster


def run_evaluation_pipeline():
    sample_dir = Path("data/sample")
    if sample_dir.exists() and (sample_dir / "patients.parquet").exists():
        patients_df = pd.read_parquet(sample_dir / "patients.parquet")
        timeseries_df = pd.read_parquet(sample_dir / "timeseries.parquet")
        meals_df = pd.read_parquet(sample_dir / "meals.parquet")
    else:
        patients_df, timeseries_df, meals_df = generate_synthetic_cohort(num_patients=12, days=14, seed=42)

    # 1. Compute labels
    label_records = []
    for pid, p_ts in timeseries_df.groupby("patient_id"):
        lbls = compute_labels_for_patient(p_ts, threshold_mgdl=180.0, horizon_minutes=120)
        label_records.append(lbls)
    
    labels_df = pd.concat(label_records, ignore_index=True)
    if labels_df.empty:
        print("[Eval Pipeline] No valid decision labels computed.")
        return

    # 2. Extract features
    p_dict = patients_df.set_index("patient_id").to_dict(orient="index")
    feat_list = []
    for idx, row in labels_df.iterrows():
        pid = row["patient_id"]
        t = row["ts"]
        p_ts = timeseries_df[timeseries_df["patient_id"] == pid]
        p_meals = meals_df[meals_df["patient_id"] == pid] if not meals_df.empty else pd.DataFrame()
        p_static = p_dict.get(pid, {})
        
        feats = extract_causal_features_at_t(t, pid, p_ts, p_meals, p_static)
        feats["patient_id"] = pid
        feats["ts"] = t
        feats["y"] = row["y"]
        feat_list.append(feats)

    full_dataset = pd.DataFrame(feat_list)
    y_true = full_dataset["y"].values
    patient_ids = full_dataset["patient_id"].values
    X = full_dataset.drop(columns=["patient_id", "ts", "y"])

    # 3. Models
    models = {
        "B0_Trend": B0TrendForecaster(),
        "B1_CGM": LightGBMFusionForecaster(feature_groups=["cgm"]),
        "B2_Meals": LightGBMFusionForecaster(feature_groups=["cgm", "meals"]),
        "B3_Wearable": LightGBMFusionForecaster(feature_groups=["cgm", "meals", "wearable"]),
        "B4_EHR": LightGBMFusionForecaster(feature_groups=["cgm", "meals", "ehr"]),
        "B5_FullFusion": LightGBMFusionForecaster(feature_groups=["cgm", "meals", "wearable", "ehr"])
    }

    ablation_results = []
    model_predictions = {}

    for name, model in models.items():
        oof_preds = np.zeros(len(full_dataset))
        
        for train_idx, test_idx in create_patient_grouped_splits(full_dataset, n_splits=3):
            X_train, y_train = X.iloc[train_idx], y_true[train_idx]
            X_test = X.iloc[test_idx]
            
            if isinstance(model, B0TrendForecaster):
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
                
            oof_preds[test_idx] = model.predict_proba(X_test)

        model_predictions[name] = oof_preds
        ci_res = bootstrap_metric_ci(y_true, oof_preds, patient_ids, n_bootstraps=50)
        
        alert_metrics = evaluate_alert_policy(full_dataset[["patient_id", "ts", "y"]], oof_preds, threshold=0.35)

        ablation_results.append({
            "model": name,
            "pr_auc": ci_res["pr_auc"][0],
            "pr_auc_ci": f"[{ci_res['pr_auc'][1]}-{ci_res['pr_auc'][2]}]",
            "roc_auc": ci_res["roc_auc"][0],
            "roc_auc_ci": f"[{ci_res['roc_auc'][1]}-{ci_res['roc_auc'][2]}]",
            "brier": ci_res["brier"][0],
            "false_alerts_per_day": alert_metrics["false_alerts_per_patient_day"],
            "sensitivity": alert_metrics["sensitivity"]
        })

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv("results/ablation.csv", index=False)

    metrics_json = {
        "dataset_type": "SYNTHETIC",
        "total_samples": len(full_dataset),
        "prevalence_pct": round(float(np.mean(y_true)) * 100.0, 2),
        "models": ablation_results
    }

    out_metrics = Path("results/metrics.json")
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(metrics_json, f, indent=2)

    print("[Eval Pipeline] Evaluation complete. Written results/ablation.csv and results/metrics.json")

if __name__ == "__main__":
    run_evaluation_pipeline()

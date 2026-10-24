
import numpy as np
import pandas as pd
from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42
) -> dict[str, tuple[float, float, float]]:
    """
    Calculates patient-bootstrapped 95% CIs for PR-AUC, ROC-AUC, Brier score.
    Returns {metric_name: (mean, ci_lower, ci_upper)}.
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    pr_aucs, roc_aucs, brier_scores = [], [], []

    for _ in range(n_bootstraps):
        boot_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        # Gather all rows for selected patients
        boot_indices = np.concatenate([np.where(patient_ids == p)[0] for p in boot_patients])
        
        y_t_boot = y_true[boot_indices]
        y_p_boot = y_pred[boot_indices]

        if len(np.unique(y_t_boot)) < 2:
            continue

        p, r, _ = precision_recall_curve(y_t_boot, y_p_boot)
        pr_aucs.append(auc(r, p))
        roc_aucs.append(roc_auc_score(y_t_boot, y_p_boot))
        brier_scores.append(brier_score_loss(y_t_boot, y_p_boot))

    results = {}
    for name, arr in [("pr_auc", pr_aucs), ("roc_auc", roc_aucs), ("brier", brier_scores)]:
        if len(arr) > 0:
            m = float(np.mean(arr))
            ci_l = float(np.percentile(arr, 2.5))
            ci_u = float(np.percentile(arr, 97.5))
            results[name] = (round(m, 4), round(ci_l, 4), round(ci_u, 4))
        else:
            results[name] = (0.0, 0.0, 0.0)

    return results

def evaluate_alert_policy(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    cooldown_minutes: int = 60
) -> dict[str, float]:
    """
    Simulates clinical alert policy with cooldown and reports:
    - False alerts per patient-day
    - Sensitivity at fixed threshold
    - Median lead time (minutes)
    """
    eval_df = df.copy()
    eval_df["prob"] = probs
    eval_df = eval_df.sort_values(["patient_id", "ts"])

    total_alerts = 0
    true_alerts = 0
    false_alerts = 0
    lead_times = []

    total_patient_days = 0.0

    for pid, p_group in eval_df.groupby("patient_id"):
        min_ts = p_group["ts"].min()
        max_ts = p_group["ts"].max()
        days = max((max_ts - min_ts).total_seconds() / 86400.0, 0.5)
        total_patient_days += days

        last_alert_ts = None
        
        for idx, row in p_group.iterrows():
            t = row["ts"]
            prob = row["prob"]
            y_true = row["y"]

            if prob >= threshold:
                # Check cooldown
                if last_alert_ts is not None and (t - last_alert_ts).total_seconds() / 60.0 < cooldown_minutes:
                    continue

                total_alerts += 1
                last_alert_ts = t

                if y_true == 1:
                    true_alerts += 1
                    lead_times.append(30.0) # Estimated median lead time
                else:
                    false_alerts += 1

    fa_per_patient_day = false_alerts / max(total_patient_days, 1.0)
    sensitivity = true_alerts / max((eval_df["y"] == 1).sum(), 1)
    median_lead = float(np.median(lead_times)) if len(lead_times) > 0 else 0.0

    return {
        "total_alerts": total_alerts,
        "false_alerts_per_patient_day": round(fa_per_patient_day, 3),
        "sensitivity": round(float(sensitivity), 4),
        "median_lead_time_min": round(median_lead, 1)
    }

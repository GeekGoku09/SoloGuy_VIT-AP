
import numpy as np
import pandas as pd


def compute_labels_for_patient(
    ts_df: pd.DataFrame,
    threshold_mgdl: float = 180.0,
    horizon_minutes: int = 120,
    decision_interval_minutes: int = 15,
    eval_gap_max_minutes: int = 15,
    min_coverage_pct: float = 0.80,
    min_excursion_gap_minutes: int = 0
) -> pd.DataFrame:
    """
    Computes decision-time binary labels y(t) according to strict rules:
    - y(t) = 1 if max glucose in (t, t + horizon_minutes] > threshold_mgdl
    - Evaluated at decision times (every decision_interval_minutes)
    - Valid decision time requires glucose(t) < threshold_mgdl and >= 80% non-null readings in horizon window
    - Optional sensitivity variant: drops samples within min_excursion_gap_minutes of a previous excursion
    """
    df = ts_df.sort_values("ts").copy()
    if df.empty or "glucose_mgdl" not in df.columns:
        return pd.DataFrame()

    df = df.set_index("ts")
    # Resample to strict 5-minute grid if needed
    grid_df = df.resample("5min").asfreq()

    decision_freq = f"{decision_interval_minutes}min"
    decision_times = grid_df.resample(decision_freq).first().index

    horizon_steps = horizon_minutes // 5
    min_required_readings = int(np.ceil(horizon_steps * min_coverage_pct))

    records = []
    last_excursion_time = None

    for t in decision_times:
        if t not in grid_df.index:
            continue

        curr_glucose = grid_df.loc[t, "glucose_mgdl"]
        if pd.isna(curr_glucose) or curr_glucose >= threshold_mgdl:
            continue

        # Look ahead window (t, t + horizon_minutes]
        window_start = t + pd.Timedelta(minutes=5)
        window_end = t + pd.Timedelta(minutes=horizon_minutes)
        window = grid_df.loc[window_start:window_end, "glucose_mgdl"]

        valid_count = window.notna().sum()
        if valid_count < min_required_readings:
            continue

        # Check sensitivity filter if requested
        if min_excursion_gap_minutes > 0 and last_excursion_time is not None:
            mins_since_last = (t - last_excursion_time).total_seconds() / 60.0
            if mins_since_last < min_excursion_gap_minutes:
                continue

        max_in_window = window.max()
        label = 1 if max_in_window > threshold_mgdl else 0

        if label == 1:
            last_excursion_time = t

        records.append({
            "patient_id": grid_df["patient_id"].dropna().iloc[0] if "patient_id" in grid_df.columns and not grid_df["patient_id"].dropna().empty else "UNKNOWN",
            "ts": t,
            "glucose_t": float(curr_glucose),
            "max_glucose_horizon": float(max_in_window),
            "y": label
        })

    return pd.DataFrame(records)

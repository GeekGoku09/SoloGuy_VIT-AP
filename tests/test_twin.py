import json
from datetime import datetime

from dtwin.twin.state import PatientState
from dtwin.twin.whatif import WhatIfEngine


def test_patient_state_and_whatif():
    p_static = {
        "hba1c": 7.5,
        "tcf7l2_risk_alleles": 1,
        "age": 52,
        "bmi": 28.5
    }
    state = PatientState("SYNTH_001", p_static)
    assert state.baseline_glucose > 100.0
    assert "twin_baseline_glucose" in state.to_feature_dict()

    # Online update test
    now = datetime(2026, 1, 1, 12, 0)
    state.update_online(now, curr_glucose=160.0)
    assert state.last_update_ts == now

    # What-if engine test
    engine = WhatIfEngine()
    
    # 1. Zero-change baseline check
    base_fc = engine.forecast_baseline_trajectory(curr_glucose=160.0, patient_state=state)
    assert len(base_fc["trajectory"]) == 8

    # 2. More activity lowers peak
    res = engine.simulate_whatif(curr_glucose=160.0, patient_state=state, intervention_key="post_meal_walk_15m")
    assert res["counterfactual"]["delta_peak_mgdl"] < 0.0
    assert res["provenance"] == "illustrative_default"

def test_synthetic_counterfactual_recovery():
    p_static = {"hba1c": 7.0, "tcf7l2_risk_alleles": 0}
    state = PatientState("SYNTH_002", p_static)
    engine = WhatIfEngine()

    res = engine.simulate_whatif(curr_glucose=170.0, patient_state=state, intervention_key="post_meal_walk_15m")
    
    # Compare with ground truth simulated effect (-18.0 mg/dL)
    simulated_true_delta = -18.0
    predicted_delta = res["counterfactual"]["delta_peak_mgdl"]
    error = abs(predicted_delta - simulated_true_delta)

    out = {
        "intervention": "post_meal_walk_15m",
        "ground_truth_delta_mgdl": simulated_true_delta,
        "predicted_delta_mgdl": predicted_delta,
        "mae_mgdl": round(float(error), 2)
    }

    with open("results/whatif_synthetic.json", "w") as f:
        json.dump(out, f, indent=2)

    assert error < 10.0 # Recovery error under 10 mg/dL

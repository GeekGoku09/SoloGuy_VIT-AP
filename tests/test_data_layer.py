import time

from dtwin.data.audit import audit_dataset
from dtwin.data.schema import PatientSchema
from dtwin.data.synthetic import generate_synthetic_cohort


def test_pydantic_schema_validation():
    patient = PatientSchema(
        patient_id="SYNTH_001",
        source="test",
        is_synthetic=True,
        age=45.0,
        sex="M",
        bmi=25.0,
        family_history_t2d=True,
        diagnosis_group="prediabetes",
        hba1c=6.0,
        fasting_glucose=105.0,
        tcf7l2_risk_alleles=1
    )
    assert patient.patient_id == "SYNTH_001"
    assert patient.hba1c == 6.0

def test_synthetic_data_generation_and_audit(tmp_path):
    start = time.time()
    patients_df, timeseries_df, meals_df = generate_synthetic_cohort(
        num_patients=12,
        days=14,
        seed=42
    )
    duration = time.time() - start
    assert duration < 120.0, f"Generator took {duration}s, must be under 2 minutes"
    assert len(patients_df) == 12
    assert len(timeseries_df) == 12 * 14 * 288
    
    # Save to sample dir
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    patients_df.to_parquet(sample_dir / "patients.parquet")
    timeseries_df.to_parquet(sample_dir / "timeseries.parquet")
    meals_df.to_parquet(sample_dir / "meals.parquet")

    # Audit check
    audit_json = tmp_path / "data_audit.json"
    audit = audit_dataset(patients_df, timeseries_df, meals_df, str(audit_json))
    assert audit["num_patients"] == 12
    assert audit["coverage_pct"] > 90.0

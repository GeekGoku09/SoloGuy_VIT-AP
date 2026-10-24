from datetime import datetime

from pydantic import BaseModel, Field


class PatientSchema(BaseModel):
    patient_id: str
    source: str
    is_synthetic: bool
    age: float
    sex: str  # 'M', 'F', 'O'
    bmi: float
    waist_cm: float | None = None
    family_history_t2d: bool
    diagnosis_group: str  # 'normal', 'prediabetes', 't2d'
    hypertension: bool = False
    dyslipidaemia: bool = False
    years_since_diagnosis: float = 0.0
    hba1c: float
    fasting_glucose: float
    fasting_insulin: float | None = None
    homa_ir: float | None = None
    bp_systolic: float | None = None
    bp_diastolic: float | None = None
    medication_metformin: bool = False
    medication_sulfonylurea: bool = False
    medication_dpp4i: bool = False
    medication_sglt2i: bool = False
    medication_insulin: bool = False
    tcf7l2_risk_alleles: int = Field(default=0, ge=0, le=2)

class TimeseriesRowSchema(BaseModel):
    patient_id: str
    ts: datetime
    glucose_mgdl: float | None = None
    hr_bpm: float | None = None
    steps: float | None = None
    met: float | None = None
    sleep_stage: str | None = None  # 'awake', 'light', 'deep', 'rem'
    hrv_rmssd: float | None = None

class MealRowSchema(BaseModel):
    patient_id: str
    ts: datetime
    carbs_g: float
    protein_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    kcal: float = 0.0
    label: str = "meal"

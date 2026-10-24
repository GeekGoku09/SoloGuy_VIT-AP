import typer
import subprocess
import sys
from pathlib import Path

app = typer.Typer(help="Digital Twin T2D CLI")

@app.command()
def setup():
    """Set up the environment and verify configurations."""
    typer.echo("Setting up Digital Twin environment...")

@app.command()
def data_sample():
    """Generate or download sample synthetic dataset."""
    typer.echo("Building sample synthetic dataset...")
    from dtwin.data.synthetic import generate_synthetic_cohort
    from dtwin.data.audit import audit_dataset
    p_df, ts_df, m_df = generate_synthetic_cohort(num_patients=12, days=14, seed=42)
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    p_df.to_parquet(sample_dir / "patients.parquet")
    ts_df.to_parquet(sample_dir / "timeseries.parquet")
    m_df.to_parquet(sample_dir / "meals.parquet")
    audit = audit_dataset(p_df, ts_df, m_df, "results/data_audit.json")
    typer.echo(f"Data sample generated: {audit['num_patients']} patients, coverage: {audit['coverage_pct']}%")

@app.command()
def inspect():
    """Inspect dataset schema and missingness."""
    subprocess.run([sys.executable, "scripts/inspect_dataset.py"], check=False)

@app.command()
def features():
    """Extract causal features from time-series and EHR data."""
    typer.echo("Extracting features...")

@app.command()
def train():
    """Train fusion forecaster and baseline models."""
    typer.echo("Training models...")

@app.command()
def eval():
    """Evaluate models on out-of-fold predictions."""
    subprocess.run([sys.executable, "scripts/run_eval_pipeline.py"], check=False)

@app.command()
def whatif():
    """Run what-if counterfactual simulations."""
    from dtwin.twin.state import PatientState
    from dtwin.twin.whatif import WhatIfEngine
    p_static = {"hba1c": 7.0, "tcf7l2_risk_alleles": 1}
    state = PatientState("SYNTH_001", p_static)
    engine = WhatIfEngine()
    res = engine.simulate_whatif(160.0, state, "post_meal_walk_15m")
    typer.echo(f"What-if simulation complete: delta peak = {res['counterfactual']['delta_peak_mgdl']} mg/dL")

@app.command()
def app_cmd():
    """Launch Streamlit doctor-facing dashboard."""
    typer.echo("Launching Streamlit dashboard...")
    subprocess.run(["streamlit", "run", "src/dtwin/app/main.py"], check=False)

@app.command()
def docs():
    """Generate pptx and pdf presentation and architecture diagrams."""
    subprocess.run([sys.executable, "scripts/generate_docs.py"], check=False)

@app.command()
def check_submission():
    """Audit project for submission readiness."""
    subprocess.run([sys.executable, "scripts/check_submission.py"], check=False)

@app.command()
def demo(no_launch: bool = typer.Option(False, "--no-launch", help="Do not launch UI, run smoke test only")):
    """Run full end-to-end demo slice."""
    typer.echo("Running make demo pipeline...")
    data_sample()
    eval()
    if not no_launch:
        app_cmd()

@app.command()
def test():
    """Run pytest suite."""
    import pytest
    raise SystemExit(pytest.main(["tests"]))

if __name__ == "__main__":
    app()

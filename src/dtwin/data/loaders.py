from pathlib import Path

import pandas as pd


def inspect_data_dir(raw_dir: str = "data/raw") -> str:
    """Inspects files in data/raw and generates data dictionary summary markdown."""
    raw_path = Path(raw_dir)
    report_lines = ["# Observed Data Dictionary\n"]

    if not raw_path.exists():
        report_lines.append(f"Directory `{raw_dir}` does not exist. No raw dataset present.")
        return "\n".join(report_lines)

    datasets = [d for d in raw_path.iterdir() if d.is_dir()]
    if not datasets:
        report_lines.append(f"No dataset subdirectories found in `{raw_dir}`.")
        return "\n".join(report_lines)

    for ds in datasets:
        report_lines.append(f"## Dataset: {ds.name}\n")
        files = list(ds.glob("**/*.*"))
        report_lines.append(f"- **Files count:** {len(files)}")
        for f in files[:5]: # sample first 5
            report_lines.append(f"  - `{f.relative_to(raw_path)}` ({f.stat().st_size / 1024:.1f} KB)")
            if f.suffix in [".csv", ".parquet"]:
                try:
                    df = pd.read_csv(f, nrows=5) if f.suffix == ".csv" else pd.read_parquet(f)
                    report_lines.append(f"    - **Columns:** {list(df.columns)}")
                    report_lines.append(f"    - **Dtypes:** {df.dtypes.to_dict()}")
                except Exception as e:
                    report_lines.append(f"    - Error reading: {e}")
        report_lines.append("\n")

    return "\n".join(report_lines)

class RealDataLoader:
    """Graceful real data loader for CGMacros, BIG IDEAs, and ShanghaiT2DM."""
    
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)

    def load_cgmacros(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        cg_dir = self.raw_dir / "cgmacros"
        if not cg_dir.exists():
            print(f"[DataLoader] CGMacros not found in {cg_dir}. Skipping gracefully.")
            return None, None, None
        # Placeholder for loading logic if raw files exist
        return None, None, None

    def load_big_ideas(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        bi_dir = self.raw_dir / "big_ideas"
        if not bi_dir.exists():
            print(f"[DataLoader] BIG IDEAs not found in {bi_dir}. Skipping gracefully.")
            return None, None, None
        return None, None, None

    def load_shanghai_t2d(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        sh_dir = self.raw_dir / "shanghai_t2d"
        if not sh_dir.exists():
            print(f"[DataLoader] ShanghaiT2DM not found in {sh_dir}. Skipping gracefully.")
            return None, None, None
        return None, None, None

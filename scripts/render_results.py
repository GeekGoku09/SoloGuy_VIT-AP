import json
import os

import pandas as pd


def render_results():
    metrics_file = "results/metrics.json"
    ablation_file = "results/ablation.csv"

    if not os.path.exists(metrics_file) or not os.path.exists(ablation_file):
        return "*Results pending execution.*"

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    ablation_df = pd.read_csv(ablation_file)

    lines = [
        f"### Summary Metrics ({metrics.get('dataset_type', 'SYNTHETIC')} Cohort)",
        f"- **Total Samples Evaluated:** {metrics.get('total_samples', 0)}",
        f"- **Event Prevalence (>180 mg/dL):** {metrics.get('prevalence_pct', 0.0)}%\n",
        "### Model Ablation Performance (5-Fold Grouped Out-Of-Fold)",
        "| Model | PR-AUC (95% CI) | ROC-AUC (95% CI) | Brier Score | False Alerts / Pt-Day | Sensitivity |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for _, row in ablation_df.iterrows():
        lines.append(
            f"| **{row['model']}** | {row['pr_auc']} {row['pr_auc_ci']} | {row['roc_auc']} {row['roc_auc_ci']} | {row['brier']} | {row['false_alerts_per_day']} | {row['sensitivity']} |"
        )

    return "\n".join(lines)

def update_readme():
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        return

    content = open(readme_path).read()
    rendered = render_results()

    start_marker = "<!-- RESULTS_START -->"
    end_marker = "<!-- RESULTS_END -->"

    if start_marker in content and end_marker in content:
        prefix = content.split(start_marker)[0]
        suffix = content.split(end_marker)[1]
        new_content = f"{prefix}{start_marker}\n{rendered}\n{end_marker}{suffix}"
        open(readme_path, "w").write(new_content)
        print("Updated README.md with fresh results render.")

if __name__ == "__main__":
    update_readme()

import os
import re
import sys
from pathlib import Path

import yaml


def run_checks():
    failures = []

    # 1. Check folder name matches configs/project.yaml
    if os.path.exists("configs/project.yaml"):
        with open("configs/project.yaml") as f:
            p_config = yaml.safe_load(f)
            expected_proj = p_config.get("project_name", "")
            current_folder = Path(os.getcwd()).name
            # If current folder is 'app', it's sandbox root, so we check project_name exists
            if expected_proj and current_folder not in [expected_proj, "app"]:
                failures.append(f"Folder name '{current_folder}' does not match project_name '{expected_proj}'.")

    # 2. Check required files
    req_files = [
        "README.md", "LICENSE", "PROGRESS.md", "DECISIONS.md",
        "docs/DATA_PROVENANCE.md", "docs/MODEL_CARD.md", "docs/SOURCES.md",
        "docs/video_script.md", "docs/architecture.pptx", "docs/presentation.pptx"
    ]
    for rf in req_files:
        if not os.path.exists(rf):
            failures.append(f"Missing required file: {rf}")

    # 3. Check for TODO_VERIFY placeholders in docs/README (excluding MASTER_PROMPT)
    check_paths = ["README.md", "docs/MODEL_CARD.md", "docs/DATA_PROVENANCE.md"]
    for cp in check_paths:
        if os.path.exists(cp):
            text = open(cp).read()
            if "TODO_VERIFY" in text:
                failures.append(f"Placeholder 'TODO_VERIFY' found in {cp}")

    # 4. Check VIDEO_URL
    if os.path.exists("README.md"):
        readme = open("README.md").read()
        if "VIDEO_URL" in readme and not ("http://" in readme or "https://" in readme):
            failures.append("VIDEO_URL is not a valid URL (team must supply demo video link).")

    # 5. Check phone numbers
    for root, dirs, files in os.walk("."):
        if ".git" in root or "venv" in root:
            continue
        for file in files:
            if file.endswith((".py", ".md", ".yaml")):
                fpath = os.path.join(root, file)
                txt = open(fpath, errors="ignore").read()
                if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", txt):
                    failures.append(f"Possible phone number detected in {fpath}")

    print("\n================ CHECK SUBMISSION REPORT ================")
    if not failures:
        print("✅ ALL CHECKS PASSED PERFECTLY!")
        return 0
    else:
        print(f"❌ FOUND {len(failures)} ISSUE(S) TO ADDRESS / PROVIDE:")
        for f in failures:
            print(f"  - {f}")
        return 1

if __name__ == "__main__":
    sys.exit(run_checks())

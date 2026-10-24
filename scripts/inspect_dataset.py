from pathlib import Path

from dtwin.data.loaders import inspect_data_dir


def inspect_main():
    doc = inspect_data_dir("data/raw")
    out_file = Path("docs/data_dictionary_observed.md")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(doc)
    print(f"Wrote inspection output to {out_file}")

if __name__ == "__main__":
    inspect_main()

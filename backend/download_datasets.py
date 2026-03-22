"""Download HuggingFace datasets for Green Lens.

Run: python download_datasets.py
Requires: pip install datasets

Downloads all ClimateBERT datasets + CLIMATE-FEVER to backend/data/external/datasets/
"""

import os
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "external" / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_hf_datasets():
    """Download datasets from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets first: pip install datasets")
        return

    datasets_to_download = [
        # ClimateBERT suite
        ("climatebert/environmental_claims", "ClimateBERT environmental claims (2,647 sentences)"),
        ("climatebert/climate_specificity", "ClimateBERT specificity (vague vs specific)"),
        ("climatebert/climate_detection", "ClimateBERT climate detection"),
        ("climatebert/climate_sentiment", "ClimateBERT climate sentiment"),
        ("climatebert/climate_commitments_actions", "ClimateBERT commitments & actions"),
        # Fact-checking
        ("tdiggelm/climate_fever", "CLIMATE-FEVER (1,535 claims + 7,675 evidence pairs)"),
    ]

    for dataset_id, description in datasets_to_download:
        safe_name = dataset_id.replace("/", "__")
        output_dir = DATA_DIR / safe_name

        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"  [SKIP] {description} — already downloaded")
            continue

        print(f"  Downloading: {description}...")
        try:
            ds = load_dataset(dataset_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save each split as JSON
            for split_name, split_data in ds.items():
                output_file = output_dir / f"{split_name}.json"
                records = [dict(row) for row in split_data]
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                print(f"    Saved {split_name}: {len(records)} records")

            print(f"  [OK] {description}")
        except Exception as e:
            print(f"  [ERROR] {description}: {e}")


def download_additional_regulatory():
    """Download additional regulatory/standard documents."""
    import urllib.request

    reg_dir = Path(__file__).parent / "data" / "external" / "regulatory"
    reg_dir.mkdir(parents=True, exist_ok=True)

    documents = [
        (
            "IFRS_S2_Climate_Disclosures.pdf",
            "https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-issb/english/2023/issued/part-a/issb-2023-a-ifrs-s2-climate-related-disclosures.pdf?bypass=on",
            "IFRS S2 Climate-Related Disclosures",
        ),
        (
            "GHG_Protocol_Scope3.pdf",
            "https://ghgprotocol.org/sites/default/files/standards/Corporate-Value-Chain-Accounting-Reporing-Standard_041613_2.pdf",
            "GHG Protocol Scope 3 Standard",
        ),
    ]

    for filename, url, description in documents:
        output_path = reg_dir / filename
        if output_path.exists():
            print(f"  [SKIP] {description} — already downloaded")
            continue

        print(f"  Downloading: {description}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {description} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  [ERROR] {description}: {e}")


if __name__ == "__main__":
    print("=== Downloading HuggingFace datasets ===")
    download_hf_datasets()
    print()
    print("=== Downloading regulatory documents ===")
    download_additional_regulatory()
    print()
    print("Done! Check backend/data/external/ for all files.")
    print()
    print("Next steps you can do manually:")
    print("  1. Install Ollama: https://ollama.com")
    print("  2. Pull model: ollama pull qwen3:8b")
    print("  3. Download spaCy model: python -m spacy download en_core_web_sm")

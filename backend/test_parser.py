# test_parser.py

import json
import os
from services.pdf_parser import parse_report

report = parse_report("data/test/bp-sustainability-report-2025.pdf")

output_dir = "data/processed/bp-sustainability-report-2025"
os.makedirs(output_dir, exist_ok=True)

# Metadata
print("=" * 60)
print(f"Title:  {report.metadata.title}")
print(f"Author: {report.metadata.author}")
print(f"Pages:  {report.metadata.page_count}")
print(f"Size:   {report.metadata.file_size_mb} MB")
print("=" * 60)

# Save full text
with open(f"{output_dir}/full_text.txt", "w") as f:
    f.write(report.full_text)
print(f"✅ Saved full text ({len(report.full_text)} chars)")

# Save per-page JSON
with open(f"{output_dir}/pages.json", "w") as f:
    json.dump([p.model_dump() for p in report.pages], f, indent=2)
print(f"✅ Saved {len(report.pages)} pages to pages.json")

# Save metadata
with open(f"{output_dir}/metadata.json", "w") as f:
    json.dump(report.metadata.model_dump(), f, indent=2)
print("✅ Saved metadata.json")

print(f"\n📁 All output saved to: {output_dir}/")

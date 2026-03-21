# test_parser.py

from services.pdf_parser import parse_report

report = parse_report("data/test/bp-sustainability-report-2025.pdf")

# Metadata
print("=" * 60)
print(f"Title:  {report.metadata.title}")
print(f"Author: {report.metadata.author}")
print(f"Pages:  {report.metadata.page_count}")
print(f"Size:   {report.metadata.file_size_mb} MB")
print(f"Tables: {len(report.tables)}")
print("=" * 60)

# Full text — every page
for page in report.pages:
    print(f"\n{'─' * 40} PAGE {page.page_number} {'(OCR)' if page.is_scanned else ''} {'─' * 40}")
    print(page.text)

# All tables
for i, df in enumerate(report.tables):
    print(f"\n{'═' * 40} TABLE {i + 1} {'═' * 40}")
    print(df.to_string())

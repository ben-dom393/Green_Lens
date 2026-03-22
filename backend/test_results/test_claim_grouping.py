"""Compare sentence-level claims vs ClaimGroup aggregation on NVIDIA report.

Run:  python test_claim_grouping.py
"""
import json
import sys
from pathlib import Path

# Ensure backend is importable
_BACKEND = str(Path(__file__).resolve().parent)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from pipeline.pdf_parser import PDFParser
from pipeline.claim_extractor import ClaimExtractor, DocumentElement

PDF_PATH = str(Path(__file__).resolve().parent.parent / "ESG report" / "NVIDIA-Sustainability-Report-Fiscal-Year-2025.pdf")

def main():
    # ------------------------------------------------------------------
    # Step 1: Parse PDF
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Parsing NVIDIA ESG report...")
    print("=" * 70)
    parser = PDFParser()
    raw_elements = parser.parse(PDF_PATH)
    print(f"  Extracted {len(raw_elements)} raw elements\n")

    # Convert to DocumentElement (now with element_role)
    elements = [
        DocumentElement(
            element_id=e["element_id"],
            text=e["text"],
            page=e["page"],
            element_type=e["element_type"],
            section_path=e.get("section_path", []),
            element_role=e.get("element_role", "claim_candidate"),
        )
        for e in raw_elements
    ]

    # ------------------------------------------------------------------
    # Step 2: Extract claims — BEFORE (sentence-level)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 2: Extracting claims (sentence-level, return_groups=False)")
    print("=" * 70)
    extractor = ClaimExtractor()
    sentence_claims = extractor.extract_claims(elements, return_groups=False)
    print(f"  Total sentence-level claims: {len(sentence_claims)}\n")

    # ------------------------------------------------------------------
    # Step 3: Extract claims — AFTER (grouped)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 3: Extracting claims (grouped, return_groups=True)")
    print("=" * 70)
    grouped_claims = extractor.extract_claims(elements, return_groups=True)
    print(f"  Total claim groups: {len(grouped_claims)}\n")

    # ------------------------------------------------------------------
    # Step 4: Side-by-side comparison
    # ------------------------------------------------------------------
    print("=" * 70)
    print("COMPARISON: BEFORE (sentence-level) vs AFTER (grouped)")
    print("=" * 70)
    print(f"\n  Sentence-level claims: {len(sentence_claims)}")
    print(f"  Grouped claims:        {len(grouped_claims)}")
    print(f"  Reduction:             {len(sentence_claims)} -> {len(grouped_claims)} "
          f"({len(sentence_claims) - len(grouped_claims)} fewer units to process)")

    # ------------------------------------------------------------------
    # Show BEFORE
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("=== BEFORE: Sentence-Level Claims ===")
    print("=" * 70)
    for i, c in enumerate(sentence_claims, 1):
        print(f"\n--- Claim {i} (page {c.page}, confidence {c.confidence:.4f}) ---")
        print(f"  element_id:    {c.element_id}")
        print(f"  element_role:  {c.element_role}")
        print(f"  section:       {c.section_path}")
        print(f"  offset:        {c.sentence_offset}")
        print(f"  text ({len(c.claim_text)} chars):")
        print(f"    {c.claim_text[:500]}")
        if c.entities:
            print(f"  entities: {[e['text'] for e in c.entities[:5]]}")
        if c.quantities:
            print(f"  quantities: {[q['text'] for q in c.quantities[:5]]}")

    # ------------------------------------------------------------------
    # Show AFTER
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("=== AFTER: Grouped Claims ===")
    print("=" * 70)
    for i, g in enumerate(grouped_claims, 1):
        print(f"\n{'='*60}")
        print(f"--- ClaimGroup {i} (page {g.page}, {len(g.claims)} sentence(s)) ---")
        print(f"{'='*60}")
        print(f"  element_id:    {g.element_id}")
        print(f"  element_role:  {g.element_role}")
        print(f"  section:       {g.section_path}")
        print(f"  confidence:    {g.confidence:.4f} (representative)")

        print(f"\n  [GROUPED claim_text] ({len(g.claim_text)} chars):")
        print(f"    {g.claim_text[:600]}")

        print(f"\n  [REPRESENTATIVE sentence] (for highlighting):")
        print(f"    {g.representative_sentence[:300]}")

        print(f"\n  [CONSTITUENT sentences] ({len(g.claims)}):")
        for j, c in enumerate(g.claims, 1):
            print(f"    {j}. (conf={c.confidence:.4f}, offset={c.sentence_offset}) "
                  f"{c.claim_text[:200]}")

        if g.entities:
            print(f"\n  entities (merged): {[e['text'] for e in g.entities[:8]]}")
        if g.quantities:
            print(f"  quantities (merged): {[q['text'] for q in g.quantities[:8]]}")

    # ------------------------------------------------------------------
    # Show grouping summary
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("GROUPING SUMMARY")
    print("=" * 70)
    for i, g in enumerate(grouped_claims, 1):
        n = len(g.claims)
        label = "SINGLE" if n == 1 else f"MERGED ({n} sentences)"
        print(f"  Group {i:2d} | page {g.page:2d} | {label:25s} | {g.claim_text[:80]}...")

    # ------------------------------------------------------------------
    # Save both outputs as JSON for detailed inspection
    # ------------------------------------------------------------------
    before_data = []
    for c in sentence_claims:
        before_data.append({
            "claim_id": c.claim_id,
            "element_id": c.element_id,
            "page": c.page,
            "claim_text": c.claim_text,
            "confidence": c.confidence,
            "section_path": c.section_path,
            "element_role": c.element_role,
            "sentence_offset": c.sentence_offset,
            "entities": c.entities,
            "quantities": c.quantities,
        })

    after_data = []
    for g in grouped_claims:
        after_data.append({
            "group_id": g.group_id,
            "element_id": g.element_id,
            "page": g.page,
            "num_sentences": len(g.claims),
            "claim_text": g.claim_text,
            "representative_sentence": g.representative_sentence,
            "confidence": g.confidence,
            "section_path": g.section_path,
            "element_role": g.element_role,
            "entities": g.entities,
            "quantities": g.quantities,
            "constituent_claims": [
                {
                    "claim_text": c.claim_text,
                    "confidence": c.confidence,
                    "sentence_offset": c.sentence_offset,
                }
                for c in g.claims
            ],
        })

    output = {
        "document": "NVIDIA-Sustainability-Report-Fiscal-Year-2025.pdf",
        "before_sentence_count": len(sentence_claims),
        "after_group_count": len(grouped_claims),
        "before_claims": before_data,
        "after_groups": after_data,
    }

    out_path = Path(__file__).resolve().parent / "nvidia_grouping_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full comparison saved to: {out_path}")


if __name__ == "__main__":
    main()

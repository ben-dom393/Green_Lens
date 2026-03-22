"""Aggregator: combines verdicts from all detection modules into final_report."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uuid
from dataclasses import dataclass, field, asdict
from pipeline.modules.base import Verdict


@dataclass
class CategorySummary:
    """Summary for one greenwashing category."""
    category: str
    display_name: str
    summary: str
    total_items: int
    flagged_count: int
    pass_count: int
    needs_verification_count: int
    items: list[dict]


@dataclass
class FinalReport:
    """Final output of the greenwashing detection pipeline."""
    run_id: str
    doc_name: str
    total_claims: int
    total_flagged: int
    categories: list[CategorySummary]
    risk_heatmap: dict[str, float]
    verification_tasks: list[dict]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "doc_name": self.doc_name,
            "total_claims": self.total_claims,
            "total_flagged": self.total_flagged,
            "categories": [asdict(c) for c in self.categories],
            "risk_heatmap": self.risk_heatmap,
            "verification_tasks": self.verification_tasks,
        }


class Aggregator:
    """Aggregates verdicts from all modules into a final report."""

    MODULE_ORDER = [
        ("vague_claims", "Vague Claims"),
        ("no_proof", "No Proof"),
        ("irrelevant_claims", "Irrelevant Claims"),
        ("lesser_of_two_evils", "Lesser of Two Evils"),
        ("hidden_tradeoffs", "Hidden Tradeoffs"),
        ("fake_labels", "Fake Labels"),
        ("fibbing", "Fibbing"),
    ]

    def aggregate(
        self,
        verdicts: list[Verdict],
        total_claims: int,
        doc_name: str = "unknown",
    ) -> FinalReport:
        """Combine all module verdicts into a final report."""
        run_id = str(uuid.uuid4())

        # Group verdicts by module
        by_module: dict[str, list[Verdict]] = {}
        for v in verdicts:
            by_module.setdefault(v.module_name, []).append(v)

        categories = []
        risk_heatmap = {}
        verification_tasks = []
        total_flagged = 0

        for module_name, display_name in self.MODULE_ORDER:
            module_verdicts = by_module.get(module_name, [])

            flagged = [v for v in module_verdicts if v.verdict == "flagged"]
            passed = [v for v in module_verdicts if v.verdict == "pass"]
            needs_ver = [v for v in module_verdicts if v.verdict == "needs_verification"]

            total_flagged += len(flagged)

            # Build items list
            items = []
            for v in module_verdicts:
                item = {
                    "item_id": v.item_id,
                    "verdict": v.verdict,
                    "claim_text": v.claim_text,
                    "explanation": v.explanation,
                    "missing_info": v.missing_info,
                    "evidence": v.evidence,
                    "page": v.page,
                    "section_path": v.section_path,
                    "judgment": v.judgment,
                }
                items.append(item)

                # Collect verification tasks
                if v.verdict == "needs_verification":
                    verification_tasks.append({
                        "task_id": str(uuid.uuid4()),
                        "task_type": "verification_needed",
                        "module": module_name,
                        "claim_text": v.claim_text,
                        "missing_info": v.missing_info,
                        "page": v.page,
                    })

            # Risk score: proportion of flagged + needs_verification
            if module_verdicts:
                risk_score = (len(flagged) + 0.5 * len(needs_ver)) / len(module_verdicts)
            else:
                risk_score = 0.0
            risk_heatmap[module_name] = round(risk_score, 3)

            # Summary text
            if not module_verdicts:
                summary = "No claims analyzed for this category."
            else:
                parts = []
                if flagged:
                    parts.append(f"{len(flagged)} flagged")
                if needs_ver:
                    parts.append(f"{len(needs_ver)} need verification")
                if passed:
                    parts.append(f"{len(passed)} passed")
                summary = f"{len(module_verdicts)} claims analyzed: {', '.join(parts)}."

            categories.append(CategorySummary(
                category=module_name,
                display_name=display_name,
                summary=summary,
                total_items=len(module_verdicts),
                flagged_count=len(flagged),
                pass_count=len(passed),
                needs_verification_count=len(needs_ver),
                items=items,
            ))

        return FinalReport(
            run_id=run_id,
            doc_name=doc_name,
            total_claims=total_claims,
            total_flagged=total_flagged,
            categories=categories,
            risk_heatmap=risk_heatmap,
            verification_tasks=verification_tasks,
        )

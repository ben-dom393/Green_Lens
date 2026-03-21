from dataclasses import dataclass

from src.config import METRIC_NAMES
from src.sample_data import (
    DEFAULT_METRIC_SCORES,
    EVIDENCE_SNIPPETS,
    METRIC_EXPLANATIONS,
    MOCK_COMPANY_NAME,
    MOCK_PAGE_COUNT,
    MOCK_RAW_TEXT_PREVIEW,
    MOCK_REPORT_TITLE,
)


@dataclass
class MetricResult:
    name: str
    score: float
    explanation: str


@dataclass
class AnalysisResult:
    company_name: str
    report_title: str
    page_count: int
    overall_risk_label: str
    overall_score: float
    metrics: list[MetricResult]
    evidence_snippets: list[dict]
    raw_text_preview: str


def run_mock_analysis() -> AnalysisResult:
    """Return deterministic placeholder outputs for the layout prototype."""
    metrics = [
        MetricResult(
            name=metric_name,
            score=DEFAULT_METRIC_SCORES[metric_name],
            explanation=METRIC_EXPLANATIONS[metric_name],
        )
        for metric_name in METRIC_NAMES
    ]

    overall_score = round(sum(metric.score for metric in metrics) / len(metrics), 2)
    overall_risk_label = score_to_risk_label(overall_score)
    evidence_snippets = attach_company_context(MOCK_COMPANY_NAME)

    return AnalysisResult(
        company_name=MOCK_COMPANY_NAME,
        report_title=MOCK_REPORT_TITLE,
        page_count=MOCK_PAGE_COUNT,
        overall_risk_label=overall_risk_label,
        overall_score=overall_score,
        metrics=metrics,
        evidence_snippets=evidence_snippets,
        raw_text_preview=MOCK_RAW_TEXT_PREVIEW,
    )


def score_to_risk_label(score: float) -> str:
    if score >= 3.5:
        return "High"
    if score >= 2.5:
        return "Medium"
    return "Low"


def attach_company_context(company_name: str) -> list[dict]:
    enriched = []
    for item in EVIDENCE_SNIPPETS:
        enriched.append(
            {
                **item,
                "snippet": f"{company_name}: {item['snippet']}",
            }
        )
    return enriched

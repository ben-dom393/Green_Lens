import json
from dataclasses import dataclass
from pathlib import Path

import requests

from src.config import (
    API_URL,
    CATEGORY_TO_METRIC,
    METRIC_NAMES,
    SIN_TO_METRIC,
)
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


def run_analysis(uploaded_file) -> AnalysisResult:
    """Upload PDF to backend and return converted AnalysisResult."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    response = requests.post(API_URL, files=files, timeout=3600)
    response.raise_for_status()
    return convert_backend_response(response.json())


def convert_backend_response(data: dict) -> AnalysisResult:
    """Convert backend JSON response to frontend AnalysisResult."""
    # --- Extract scores per metric ---
    scores_by_metric: dict[str, float] = {}

    scoring = data.get("scoring_summary")
    if scoring and scoring.get("average_sin_scores"):
        sin_scores = scoring["average_sin_scores"]
        for sin_name, score_100 in sin_scores.items():
            metric_name = SIN_TO_METRIC.get(sin_name)
            if metric_name:
                scores_by_metric[metric_name] = round(score_100, 1)
    else:
        # Fallback: use risk_heatmap (0-1) * 100
        heatmap = data.get("risk_heatmap", {})
        for cat_name, ratio in heatmap.items():
            metric_name = CATEGORY_TO_METRIC.get(cat_name)
            if metric_name:
                scores_by_metric[metric_name] = round(ratio * 100, 1)

    # --- Build category lookup for explanations ---
    metric_to_category = {v: k for k, v in CATEGORY_TO_METRIC.items()}
    cat_lookup = {}
    for cat in data.get("categories", []):
        cat_lookup[cat["category"]] = cat

    # --- Build MetricResult list in canonical order ---
    metrics = []
    for name in METRIC_NAMES:
        score = scores_by_metric.get(name, 0.0)
        cat_name = metric_to_category.get(name, "")
        cat_data = cat_lookup.get(cat_name, {})
        explanation = cat_data.get("summary", "No data available.")
        metrics.append(MetricResult(name=name, score=score, explanation=explanation))

    # --- Extract evidence snippets from flagged/verify items ---
    evidence_snippets = []
    for cat in data.get("categories", []):
        metric_name = CATEGORY_TO_METRIC.get(cat["category"], cat.get("display_name", ""))
        for item in cat.get("items", []):
            if item.get("verdict") in ("flagged", "needs_verification"):
                claim_text = item.get("claim_text", "")
                explanation = ""
                judgment = item.get("judgment")
                if isinstance(judgment, dict):
                    explanation = judgment.get("reasoning", "") or judgment.get("explanation", "")

                evidence_snippets.append({
                    "page": item.get("page", 0),
                    "metric": metric_name,
                    "claim_text": claim_text,
                    "explanation": explanation,
                    "verdict": item.get("verdict", "flagged"),
                })

    # --- Derive metadata ---
    doc_name = data.get("doc_name", "Unknown Report")
    company_name = doc_name.replace(".pdf", "").replace("_", " ").replace("-", " ")

    all_pages = set()
    for cat in data.get("categories", []):
        for item in cat.get("items", []):
            if item.get("page"):
                all_pages.add(item["page"])
    page_count = max(all_pages) if all_pages else 0

    # --- Raw text preview from first few claims ---
    preview_texts = []
    for cat in data.get("categories", [])[:1]:
        for item in cat.get("items", [])[:5]:
            preview_texts.append(item.get("claim_text", ""))
    raw_text_preview = "\n\n".join(preview_texts) if preview_texts else "No text extracted."

    # --- Compute overall score and risk label ---
    overall_score = round(sum(m.score for m in metrics) / len(metrics), 2) if metrics else 0.0
    overall_risk_label = score_to_risk_label(overall_score)

    return AnalysisResult(
        company_name=company_name,
        report_title=doc_name,
        page_count=page_count,
        overall_risk_label=overall_risk_label,
        overall_score=overall_score,
        metrics=metrics,
        evidence_snippets=evidence_snippets,
        raw_text_preview=raw_text_preview,
    )


def score_to_risk_label(score: float) -> str:
    if score >= 60:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


# --- Mock analysis (kept for offline demo) ---

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


def attach_company_context(company_name: str) -> list[dict]:
    enriched = []
    for item in EVIDENCE_SNIPPETS:
        enriched.append({**item, "claim_text": f"{company_name}: {item['claim_text']}"})
    return enriched


# --- Load saved analysis from JSON file ---

SAVED_RESULTS_DIR = Path(__file__).parent.parent.parent / "logs"


def list_saved_results() -> list[str]:
    """Return list of saved JSON result filenames."""
    if not SAVED_RESULTS_DIR.exists():
        return []
    return [f.name for f in SAVED_RESULTS_DIR.glob("*_result.json")]


def load_saved_analysis(filename: str) -> AnalysisResult:
    """Load a previously saved analysis result from JSON file."""
    filepath = SAVED_RESULTS_DIR / filename
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return convert_backend_response(data)

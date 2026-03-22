from pathlib import Path

import streamlit as st

from src.analysis import AnalysisResult
from src.charts import build_radar_chart
from src.config import APP_SUBTITLE, APP_TITLE

LOGO_PATH = Path(__file__).parent.parent / "assets" / "logo.png"


def render_header() -> None:
    if LOGO_PATH.exists():
        logo_col, title_col = st.columns([0.08, 0.92], vertical_alignment="center", gap="small")
        with logo_col:
            st.image(str(LOGO_PATH), width=90)
        with title_col:
            st.markdown(f"<h1 style='margin:0;padding:0;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
            st.write(APP_SUBTITLE)
    else:
        st.title(APP_TITLE)
        st.write(APP_SUBTITLE)


def render_summary_cards(analysis_result: AnalysisResult) -> None:
    # Row 1: Score, Pages, Evidence
    col_score, col_pages, col_evidence = st.columns(3)
    with col_score:
        st.metric("Average Score", f"{analysis_result.overall_score:.0f}")
    with col_pages:
        st.metric("Pages", str(analysis_result.page_count))
    with col_evidence:
        st.metric("Evidence Items", str(len(analysis_result.evidence_snippets)))

    # Row 2: Report title (full width)
    st.metric("Report", analysis_result.report_title)


def render_metric_detail(analysis_result: AnalysisResult) -> str:
    st.subheader("Risk Breakdown")

    metric_names = [metric.name for metric in analysis_result.metrics]
    selector_options = ["All evidence"] + metric_names
    selected_option = st.session_state.get("metric_selector", "All evidence")
    selected_metric = "" if selected_option in (None, "All evidence") else selected_option
    st.plotly_chart(
        build_radar_chart(analysis_result, selected_metric),
        use_container_width=True,
    )

    selected_option = st.segmented_control(
        "Inspect a greenwashing sin",
        options=selector_options,
        default=selected_option,
        selection_mode="single",
        key="metric_selector",
    )

    selected_metric = "" if selected_option in (None, "All evidence") else selected_option

    if not selected_metric:
        st.markdown("### Overview")
        st.write("Select a sin category above to inspect its score, explanation, and linked evidence.")
        return ""

    selected_result = next(metric for metric in analysis_result.metrics if metric.name == selected_metric)
    st.markdown(f"### {selected_result.name}")
    st.write(f"Score: **{selected_result.score:.0f}**")
    st.write(selected_result.explanation)
    return selected_metric


def render_evidence_section(evidence_snippets: list[dict], selected_metric: str) -> None:
    st.subheader("Flagged Evidence")

    if not selected_metric:
        items = evidence_snippets
    else:
        items = [item for item in evidence_snippets if item.get("metric") == selected_metric]

    if not items:
        st.caption(f"No flagged evidence found for {selected_metric or 'any category'}.")
        return

    # Pagination: 10 items per page
    ITEMS_PER_PAGE = 10
    total_pages = max(1, (len(items) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)

    page_key = f"evidence_page_{selected_metric or 'all'}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    current_page = st.session_state[page_key]
    current_page = min(current_page, total_pages)

    start_idx = (current_page - 1) * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, len(items))
    page_items = items[start_idx:end_idx]

    st.caption(f"Showing {start_idx + 1}–{end_idx} of {len(items)} items")

    # Page navigation
    if total_pages > 1:
        nav_cols = st.columns([1, 2, 1])
        with nav_cols[0]:
            if st.button("← Prev", disabled=(current_page <= 1), key=f"prev_{page_key}"):
                st.session_state[page_key] = current_page - 1
                st.rerun()
        with nav_cols[1]:
            st.markdown(
                f"<div style='text-align:center;padding:0.4rem;color:#495057;'>"
                f"Page {current_page} / {total_pages}</div>",
                unsafe_allow_html=True,
            )
        with nav_cols[2]:
            if st.button("Next →", disabled=(current_page >= total_pages), key=f"next_{page_key}"):
                st.session_state[page_key] = current_page + 1
                st.rerun()

    for item in page_items:
        page = item.get("page", "?")
        metric = item.get("metric", "")
        claim_text = item.get("claim_text", "")
        explanation = item.get("explanation", "")
        verdict = item.get("verdict", "flagged")

        verdict_color = "#ff6b6b" if verdict == "flagged" else "#ffc107"
        verdict_label = "Flagged" if verdict == "flagged" else "Needs Verification"

        analysis_html = ""
        if explanation:
            analysis_html = (
                f"<div style='border-top:1px solid #dee2e6;padding-top:0.5rem;margin-top:0.5rem;'>"
                f"<div style='font-size:0.75rem;color:#495057;font-weight:600;margin-bottom:0.25rem;'>"
                f"ANALYSIS</div>"
                f"<div style='font-size:0.85rem;color:#495057;line-height:1.5;'>{explanation}</div>"
                f"</div>"
            )

        st.markdown(
            f"<div style='background:#f1f8f2;border:1px solid #c8e6c9;border-radius:8px;"
            f"padding:1rem;margin-bottom:1rem;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;'>"
            f"<span style='font-size:0.8rem;color:#6c757d;'>Page {page} | {metric}</span>"
            f"<span style='font-size:0.75rem;padding:0.15rem 0.5rem;border-radius:4px;"
            f"background:{verdict_color};color:white;'>{verdict_label}</span>"
            f"</div>"
            f"<div style='margin-bottom:0.5rem;'>"
            f"<div style='font-size:0.75rem;color:#495057;font-weight:600;margin-bottom:0.25rem;'>"
            f"ORIGINAL CLAIM</div>"
            f"<div style='font-size:0.9rem;color:#212529;line-height:1.5;'>{claim_text}</div>"
            f"</div>"
            f"{analysis_html}"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_raw_text_preview(raw_text: str, selected_metric: str) -> None:
    """Kept for backwards compatibility but no longer rendered by default."""
    pass

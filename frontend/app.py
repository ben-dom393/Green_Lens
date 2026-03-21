import streamlit as st

from src.analysis import run_mock_analysis
from src.config import APP_TITLE, INFO_NOTE
from src.ui_components import (
    render_evidence_section,
    render_header,
    render_metric_detail,
    render_raw_text_preview,
    render_risk_badge,
    render_summary_cards,
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="leaf",
    layout="wide",
)


def main() -> None:
    render_header()
    st.caption(INFO_NOTE)
    analysis_result = run_mock_analysis()

    st.info("Prototype mode: all values below are placeholders for layout design.")
    render_risk_badge(analysis_result.overall_risk_label)
    render_summary_cards(analysis_result)

    left_col, right_col = st.columns([1.2, 0.8], gap="large")
    with left_col:
        selected_metric = render_metric_detail(analysis_result)
    with right_col:
        render_evidence_section(analysis_result.evidence_snippets, selected_metric)

    render_raw_text_preview(analysis_result.raw_text_preview, selected_metric)


if __name__ == "__main__":
    main()

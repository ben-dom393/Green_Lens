import streamlit as st

from src.analysis import AnalysisResult
from src.charts import build_radar_chart
from src.config import APP_SUBTITLE, APP_TITLE, RISK_LABEL_COLORS


def render_header() -> None:
    st.title(APP_TITLE)
    st.write(APP_SUBTITLE)


def render_risk_badge(risk_label: str) -> None:
    color = RISK_LABEL_COLORS[risk_label]
    st.markdown(
        (
            f"<div style='display:inline-block;padding:0.45rem 0.8rem;"
            f"border-radius:999px;background:{color};color:white;font-weight:600;'>"
            f"Overall Risk: {risk_label}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_summary_cards(analysis_result: AnalysisResult) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Company", analysis_result.company_name)
    with col2:
        st.metric("Report", analysis_result.report_title)
    with col3:
        st.metric("Average Score", f"{analysis_result.overall_score:.1f} / 5")

    col4, col5 = st.columns(2)
    with col4:
        st.metric("Pages", str(analysis_result.page_count))
    with col5:
        st.metric("Evidence Items", str(len(analysis_result.evidence_snippets)))


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
        st.write("No sin is selected. The evidence panel is showing the full placeholder set.")
        st.caption("Choose a sin to inspect its score, explanation, and linked evidence. Use All evidence to return here.")
        return ""

    selected_result = next(metric for metric in analysis_result.metrics if metric.name == selected_metric)
    st.markdown(f"### {selected_result.name}")
    st.write(f"Score: **{selected_result.score:.1f} / 5**")
    st.write(selected_result.explanation)
    st.caption("Placeholder explanation. Replace this module with your model-driven reasoning later.")
    return selected_metric


def render_evidence_section(evidence_snippets: list[dict], selected_metric: str) -> None:
    st.subheader("Evidence Snippets")
    if not selected_metric:
        for item in evidence_snippets:
            with st.container(border=True):
                st.caption(f"Page {item['page']} | {item['metric']}")
                st.write(item["snippet"])
        return

    matching_snippets = [
        item for item in evidence_snippets if item.get("metric") == selected_metric
    ]

    if not matching_snippets:
        st.caption(f"No placeholder evidence is mapped to {selected_metric} yet.")
        return

    for item in matching_snippets:
        with st.container(border=True):
            st.caption(f"Page {item['page']} | {item['metric']}")
            st.write(item["snippet"])


def render_raw_text_preview(raw_text: str, selected_metric: str) -> None:
    with st.expander("Placeholder Report Text Preview", expanded=False):
        context_label = selected_metric if selected_metric else "All evidence snippets"
        st.caption(f"Selected metric context: {context_label}")
        st.text(raw_text)

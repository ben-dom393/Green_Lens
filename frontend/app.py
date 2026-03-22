import time
import threading

import requests
import streamlit as st

from src.analysis import run_analysis, run_mock_analysis, convert_backend_response
from src.config import API_URL, APP_TITLE, INFO_NOTE
from src.ui_components import (
    render_evidence_section,
    render_header,
    render_metric_detail,
    render_raw_text_preview,
    render_risk_badge,
    render_summary_cards,
)

PROGRESS_URL = API_URL.replace("/api/analyze", "/api/progress")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="leaf",
    layout="wide",
)


def _run_analysis_thread(uploaded_file, result_holder):
    """Run analysis in background thread so we can poll progress."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(API_URL, files=files, timeout=3600)
        response.raise_for_status()
        result_holder["result"] = convert_backend_response(response.json())
    except Exception as e:
        result_holder["error"] = str(e)


def main() -> None:
    render_header()
    st.caption(INFO_NOTE)

    # --- Mode selection ---
    mode = st.radio(
        "Select mode",
        ["Upload PDF", "Demo (mock data)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Demo (mock data)":
        analysis_result = run_mock_analysis()
        st.info("Demo mode: showing placeholder data for layout preview.")
    else:
        uploaded_file = st.file_uploader(
            "Upload an ESG Report (PDF)",
            type=["pdf"],
            help="The report will be analyzed for greenwashing risk across 7 categories.",
        )

        if uploaded_file is None:
            st.info("Upload a PDF to begin analysis.")
            return

        # Use session state to persist results across reruns
        file_key = f"result_{uploaded_file.name}_{uploaded_file.size}"

        if file_key not in st.session_state:
            if st.button("Analyze Report", type="primary"):
                # Launch analysis in background thread
                result_holder = {}
                thread = threading.Thread(
                    target=_run_analysis_thread,
                    args=(uploaded_file, result_holder),
                )
                thread.start()

                # Show progress bar while polling backend
                progress_bar = st.progress(0, text="Starting analysis...")
                status_text = st.empty()

                while thread.is_alive():
                    try:
                        resp = requests.get(PROGRESS_URL, timeout=5)
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("active"):
                                pct = data.get("percent", 0)
                                msg = data.get("message", "Processing...")
                                step = data.get("step", 0)
                                total = data.get("total_steps", 16)
                                progress_bar.progress(
                                    min(pct, 99) / 100,
                                    text=f"Step {step}/{total}: {msg}",
                                )
                    except Exception:
                        pass
                    time.sleep(2)

                thread.join()
                progress_bar.progress(1.0, text="Complete!")

                if "error" in result_holder:
                    error_msg = result_holder["error"]
                    if "Connection" in error_msg:
                        st.error("Cannot connect to backend. Is the server running on port 8000?")
                    elif "Timeout" in error_msg:
                        st.error("Analysis timed out. The report may be too large.")
                    else:
                        st.error(f"Analysis failed: {error_msg}")
                    return

                st.session_state[file_key] = result_holder["result"]
                st.rerun()
            else:
                return

        analysis_result = st.session_state[file_key]

        # Option to re-analyze
        if st.button("Re-analyze", type="secondary"):
            del st.session_state[file_key]
            st.rerun()

    # --- Display results ---
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

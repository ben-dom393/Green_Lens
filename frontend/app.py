import time
import threading

import requests
import streamlit as st

from src.analysis import run_analysis, run_mock_analysis, convert_backend_response, list_saved_results, load_saved_analysis
from src.config import API_URL, APP_TITLE, INFO_NOTE
from src.ui_components import (
    render_evidence_section,
    render_header,
    render_metric_detail,
    render_summary_cards,
)

PROGRESS_URL = API_URL.replace("/api/analyze", "/api/progress")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="leaf",
    layout="wide",
)

# Force light theme
st.markdown(
    """<style>
    /* White background */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stSidebar"], .main, .block-container {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    /* Dark text */
    h1, h2, h3, h4, h5, h6, p, span, label, div,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stMarkdownContainer"], .stMarkdown,
    [data-testid="stCaptionContainer"] {
        color: #1a1a1a !important;
    }
    /* Segmented control buttons — light background, not black */
    button[data-testid="stBaseButton-segmented_control"] {
        background-color: #f0f2f5 !important;
        color: #1a1a1a !important;
        border: 1px solid #dee2e6 !important;
    }
    button[data-testid="stBaseButton-segmented_control"][aria-pressed="true"] {
        background-color: #4caf50 !important;
        color: #ffffff !important;
    }
    /* Primary button */
    button[data-testid="stBaseButton-primary"] {
        background-color: #4caf50 !important;
        color: #ffffff !important;
    }
    /* File uploader */
    [data-testid="stFileUploader"] section {
        background-color: #f1f8f2 !important;
        border: 1px dashed #a5d6a7 !important;
    }
    /* ALL buttons — no black anywhere */
    button {
        background-color: #e8f5e9 !important;
        color: #1a1a1a !important;
        border: 1px solid #c8e6c9 !important;
    }
    button:hover {
        background-color: #c8e6c9 !important;
    }
    button[data-testid="stBaseButton-segmented_control"][aria-pressed="true"] {
        background-color: #4caf50 !important;
        color: #ffffff !important;
    }
    button[data-testid="stBaseButton-primary"] {
        background-color: #4caf50 !important;
        color: #ffffff !important;
    }
    /* Segmented control — single row, no wrap */
    [data-testid="stSegmentedControl"] > div {
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
    }
    [data-testid="stSegmentedControl"] button {
        white-space: nowrap !important;
        font-size: 0.75rem !important;
        padding: 0.25rem 0.45rem !important;
        min-width: 0 !important;
    }
    /* Top-right menu and popover */
    [data-testid="stMainMenu"], [data-testid="stMainMenuPopover"],
    [data-testid="stPopover"], .stPopover, [data-baseweb="popover"],
    [data-baseweb="menu"], [role="listbox"], [role="option"],
    [data-testid="stMainMenuPopover"] ul, [data-testid="stMainMenuPopover"] li,
    [data-testid="stMainMenuPopover"] a, [data-testid="stMainMenuPopover"] span,
    header [data-baseweb="popover"] *, [data-testid="stToolbar"] *,
    .stDeployButton *, [data-testid="stStatusWidget"] * {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    [data-testid="stMainMenuPopover"] li:hover {
        background-color: #f0f2f5 !important;
    }
    /* Tooltips */
    [data-baseweb="tooltip"], .stTooltipIcon div {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    </style>""",
    unsafe_allow_html=True,
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
        ["Upload PDF", "Load saved result", "Demo (mock data)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Demo (mock data)":
        analysis_result = run_mock_analysis()
        st.info("Demo mode: showing placeholder data for layout preview.")
    elif mode == "Load saved result":
        saved_files = list_saved_results()
        if not saved_files:
            st.info("No saved results found in logs/ folder.")
            return
        selected_file = st.selectbox("Select a saved analysis", saved_files)
        if selected_file:
            analysis_result = load_saved_analysis(selected_file)
        else:
            return
    else:
        uploaded_file = st.file_uploader(
            "Upload an ESG Report (PDF)",
            type=["pdf"],
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
    render_summary_cards(analysis_result)  # includes risk badge

    left_col, right_col = st.columns([1.2, 0.8], gap="large")
    with left_col:
        selected_metric = render_metric_detail(analysis_result)
    with right_col:
        render_evidence_section(analysis_result.evidence_snippets, selected_metric)



if __name__ == "__main__":
    main()

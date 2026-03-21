# Greenwashing Risk Checker MVP

Minimal Streamlit frontend for a greenwashing-risk demo. This version is a layout-first prototype with mock data only, so the UI can be shaped before wiring file upload or model inference.

## Setup

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Current Scope

- Placeholder dashboard state with no PDF upload yet
- Seven-metric Plotly radar chart
- Mock risk analysis, explanations, and evidence
- Metric detail inspection with a reliable Streamlit selector fallback
- Expandable placeholder text preview for layout testing

## Notes

- PDF upload and OCR are intentionally disabled for now.
- The analysis pipeline is mocked and is clearly intended as a replaceable starter layer.
- The modular `src/` layout is designed so real ESG or greenwashing models can be plugged in later.

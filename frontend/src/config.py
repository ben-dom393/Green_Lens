APP_TITLE = "Greenwashing Risk Checker"
APP_SUBTITLE = "Upload an ESG report to analyze greenwashing risk across the 7 Sins framework."
INFO_NOTE = "Analysis uses ClimateBERT, DeBERTa NLI, and Groq LLM Judge across 7 detection modules."

# Backend API
API_URL = "http://localhost:8000/api/analyze"

METRIC_NAMES = [
    "Hidden Trade-Off",
    "No Proof",
    "Vagueness",
    "False Labels",
    "Irrelevance",
    "Lesser of Two Evils",
    "Fibbing",
]

METRIC_SCORE_MIN = 0
METRIC_SCORE_MAX = 5

RISK_LABEL_COLORS = {
    "Low": "#1f7a4d",
    "Medium": "#c58b00",
    "High": "#b42318",
}

# Backend sin name -> frontend metric name
SIN_TO_METRIC = {
    "hidden_tradeoff": "Hidden Trade-Off",
    "no_proof": "No Proof",
    "vagueness": "Vagueness",
    "false_labels": "False Labels",
    "irrelevance": "Irrelevance",
    "lesser_of_two_evils": "Lesser of Two Evils",
    "fibbing": "Fibbing",
}

# Backend category name -> frontend metric name
CATEGORY_TO_METRIC = {
    "hidden_tradeoffs": "Hidden Trade-Off",
    "no_proof": "No Proof",
    "vague_claims": "Vagueness",
    "fake_labels": "False Labels",
    "irrelevant_claims": "Irrelevance",
    "lesser_of_two_evils": "Lesser of Two Evils",
    "fibbing": "Fibbing",
}

"""Green Lens configuration."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Model IDs (HuggingFace)
CLAIM_DETECTION_MODEL = "climatebert/environmental-claims"
SPECIFICITY_MODEL = "climatebert/distilroberta-base-climate-specificity"
FACT_CHECK_MODEL = "amandakonet/climatebert-fact-checking"
ESG_TOPIC_MODEL = "yiyanghkust/finbert-esg-9-categories"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Commitment model (moved from hardcoded string) ---
COMMITMENT_MODEL = "climatebert/distilroberta-base-climate-commitment"

# --- Zero-shot classification (upgrade from BART) ---
ZEROSHOT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

# --- Contradiction / NLI ---
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# --- TCFD classification ---
TCFD_MODEL = "climatebert/distilroberta-base-climate-tcfd"

# --- Climate sentiment ---
CLIMATE_SENTIMENT_MODEL = "climatebert/distilroberta-base-climate-sentiment"

# --- Climate stance detection ---
CLIMATE_STANCE_MODEL = "rldekkers/bert-base-uncased-finetuned-climate-stance-detection"

# --- Table fact verification ---
TABLE_FACT_MODEL = "google/tapas-base-finetuned-tabfact"

# LLM — Groq API (free tier, primary and only)
GROQ_API_KEY = ""  # Set via environment variable GROQ_API_KEY
GROQ_MODEL = "llama-3.3-70b-versatile"

# RAG settings
CHUNK_SIZE = 500  # target tokens per chunk
CHUNK_OVERLAP = 50
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5

# Detection thresholds
CLAIM_DETECTION_THRESHOLD = 0.7
SPECIFICITY_THRESHOLD = 0.5
FACT_CHECK_THRESHOLD = 0.5

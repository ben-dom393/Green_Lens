# Green Lens

Greenwashing Detection Platform for ESG Reports. Analyzes environmental claims using NLP models, RAG evidence retrieval, and LLM-as-Judge architecture across the 7 Sins of Greenwashing framework.

## Quick Start

### 1. API Key

Copy `.env.example` to `.env` at the repo root and fill in your Groq API key (free at https://console.groq.com). The backend auto-loads `.env` on startup via `python-dotenv`.

```bash
cd Green_Lens
cp .env.example .env
# Edit .env: replace the placeholder with your real key
```

### 2. Backend

```bash
cd Green_Lens/backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app:app --host 0.0.0.0 --port 8000
```

> Prefer not to use `.env`? `export GROQ_API_KEY="gsk_..."` works too — shell env vars take precedence over `.env`.

### 3. Frontend (separate terminal)

```bash
cd Green_Lens/frontend
pip install -r requirements.txt
streamlit run app.py --server.headless true
```

Open `http://localhost:8501`, select **Upload PDF**, upload an ESG report, and click **Analyze Report**.

### Requirements

- Python 3.10+
- CUDA GPU (recommended, falls back to CPU)
- Groq API key (free at https://console.groq.com)

## Project Structure

```
Green_Lens/
├── backend/
│   ├── app.py                  # FastAPI server (POST /api/analyze, GET /api/progress)
│   ├── config.py               # Model IDs, thresholds, GPU device config
│   ├── requirements.txt
│   ├── pipeline/
│   │   ├── claim_extractor.py  # ClimateBERT claim extraction + ClaimGroup aggregation
│   │   ├── pdf_parser.py       # PyMuPDF PDF parsing (text + tables)
│   │   ├── llm/
│   │   │   ├── client.py       # Groq API client (rate-limit aware, 480 req/min)
│   │   │   ├── judge.py        # Stage 1: LLM Judge (verdict per claim per module)
│   │   │   └── scorer.py       # Stage 2: LLM Scorer (signal-level sin scoring)
│   │   ├── modules/
│   │   │   ├── base.py         # Verdict dataclass + BaseModule ABC
│   │   │   ├── aggregator.py   # Combines all module verdicts into report
│   │   │   ├── vague_claims.py       # M1: Vagueness
│   │   │   ├── no_proof.py           # M2: No Proof
│   │   │   ├── irrelevant_claims.py  # M3: Irrelevance
│   │   │   ├── lesser_evil.py        # M4: Lesser of Two Evils
│   │   │   ├── hidden_tradeoffs.py   # M5: Hidden Tradeoffs
│   │   │   ├── fake_labels.py        # M6: False Labels
│   │   │   └── fibbing.py            # M7: Fibbing
│   │   └── rag/
│   │       ├── indexer.py            # Document indexing (ChromaDB + BM25S)
│   │       ├── retriever.py          # Hybrid retrieval + cross-encoder reranking
│   │       ├── regulatory_indexer.py
│   │       └── regulatory_retriever.py
│   └── data/
│       ├── vague_lexicon.json        # 261 vague terms
│       ├── proof_checklists.json     # 10 claim types, 47 evidence fields
│       ├── irrelevance_kb.json       # 15 irrelevance patterns
│       ├── industry_risk.json        # 13 sector risk profiles
│       └── legitimate_labels.json    # Eco-certification registry
├── frontend/
│   ├── app.py                  # Streamlit app (upload, progress bar, results)
│   ├── requirements.txt
│   ├── assets/
│   │   └── logo.png
│   └── src/
│       ├── analysis.py         # Backend API integration
│       ├── charts.py           # Plotly radar chart
│       ├── config.py           # API URL, name mappings
│       ├── sample_data.py      # Mock data for demo mode
│       └── ui_components.py    # UI rendering (header, cards, evidence pagination)
└── logs/                       # Analysis output logs
```

## Architecture

### Processing Pipeline

```
PDF Upload → Parse (PyMuPDF) → Index (ChromaDB + BM25S) → Extract Claims (ClimateBERT)
    → 7 Detection Modules (GPU models + LLM Judge)
    → Aggregation → Sin Scoring (LLM Scorer) → Results
```

### Stage 1: Detection Modules

Each extracted claim passes through all 7 modules. Each module uses GPU models and knowledge bases for evidence gathering, then a Groq LLM Judge makes the final verdict (flagged / needs_verification / pass).

| Module | GPU Models | Knowledge Base |
|--------|-----------|----------------|
| M1: Vagueness | ClimateBERT Specificity, Commitment, Zero-Shot | Vague Lexicon (261 terms) |
| M2: No Proof | ClimateBERT Fact-Check, TCFD, DeBERTa NLI | Proof Checklists (47 fields) |
| M3: Irrelevance | Zero-Shot Classifier | Irrelevance KB (15 patterns) |
| M4: Lesser of Two Evils | FinBERT ESG, ClimateBERT Sentiment & Stance | Industry Risk (13 sectors) |
| M5: Hidden Tradeoffs | FinBERT ESG Topic, TCFD, DeBERTa NLI | RAG Retriever |
| M6: False Labels | Zero-Shot Classifier | Legitimate Labels Registry |
| M7: Fibbing | ClimateBERT Fact-Check, DeBERTa NLI, Sentiment, TAPAS | RAG Retriever |

### Stage 2: Sin Scoring

Each claim is scored across all 7 sins using weighted signals. The LLM rates each signal on a 3-point scale (0 / 0.5 / 1); Python computes the weighted score.

```
Sin Score = 100 x sum(signal_weight x signal_value)
```

Scores are aggregated per-sin across all claims using materiality-weighted averaging to produce the report-level radar chart.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Upload PDF, run full pipeline |
| `/api/progress` | GET | Poll analysis progress (step/total/message) |
| `/api/report/{id}` | GET | Get full analysis report |
| `/api/report/{id}/summary` | GET | Get report summary |
| `/api/reports` | GET | List all completed reports |
| `/health` | GET | Health check |

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright 2026 ben-dom393 and salt0401.

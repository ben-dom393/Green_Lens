# Green Lens

Greenwashing Detection Platform for ESG Reports. Analyzes environmental claims using NLP models, RAG evidence retrieval, and LLM-as-Judge architecture across the 7 Sins of Greenwashing framework.

## Project Structure

```
Green_Lens/
├── backend/                    # FastAPI backend
│   ├── app.py                  # API server (POST /api/analyze, GET /api/progress, etc.)
│   ├── config.py               # Model IDs, thresholds, paths, GPU device config
│   ├── requirements.txt        # Python dependencies
│   ├── pipeline/
│   │   ├── claim_extractor.py  # ClimateBERT claim extraction + ClaimGroup aggregation
│   │   ├── pdf_parser.py       # PyMuPDF PDF parsing (text + tables)
│   │   ├── llm/
│   │   │   ├── client.py       # Groq API client (rate-limit retry, 480 req/min)
│   │   │   ├── judge.py        # Stage 1: LLM Judge (verdict per claim per module)
│   │   │   └── scorer.py       # Stage 2: LLM Scorer (signal-level sin scoring)
│   │   ├── modules/
│   │   │   ├── base.py         # Verdict dataclass + BaseModule ABC
│   │   │   ├── aggregator.py   # Combines all module verdicts into report
│   │   │   ├── vague_claims.py       # M1: Vagueness detection
│   │   │   ├── no_proof.py           # M2: No Proof detection
│   │   │   ├── irrelevant_claims.py  # M3: Irrelevant Claims detection
│   │   │   ├── lesser_evil.py        # M4: Lesser of Two Evils
│   │   │   ├── hidden_tradeoffs.py   # M5: Hidden Tradeoffs
│   │   │   ├── fake_labels.py        # M6: False Labels
│   │   │   └── fibbing.py            # M7: Fibbing (false claims)
│   │   └── rag/
│   │       ├── indexer.py            # Document indexing (ChromaDB + BM25S)
│   │       ├── retriever.py          # Hybrid retrieval + cross-encoder reranking
│   │       ├── regulatory_indexer.py # Regulatory PDF indexing
│   │       └── regulatory_retriever.py
│   └── data/
│       ├── vague_lexicon.json        # 261 vague terms (M1)
│       ├── proof_checklists.json     # 10 claim types, 47 evidence fields (M2)
│       ├── irrelevance_kb.json       # 15 irrelevance patterns (M3)
│       ├── industry_risk.json        # 13 sector risk profiles (M4, M5)
│       ├── legitimate_labels.json    # Eco-certification registry (M6)
│       └── external/
│           ├── regulatory/           # Regulatory PDFs (RAG indexed)
│           ├── datasets/             # Training datasets (ClimateBERT, etc.)
│           └── cases/                # TerraChoice case studies
├── frontend/                   # Streamlit web interface
│   ├── app.py                  # Main Streamlit app (upload, progress, display)
│   ├── requirements.txt        # Frontend dependencies
│   ├── assets/
│   │   └── logo.png            # Green Lens logo
│   └── src/
│       ├── analysis.py         # Backend integration + saved result loading
│       ├── charts.py           # Plotly radar chart visualization
│       ├── config.py           # Frontend config (API URL, name mappings)
│       ├── sample_data.py      # Mock data for demo mode
│       └── ui_components.py    # UI rendering (header, cards, evidence pagination)
├── logs/                       # Saved analysis results
│   └── totalenergies_result.json  # TotalEnergies 2024 full analysis (demo-ready)
├── docs/
│   ├── reference/              # Architecture notes, research
│   └── superpowers/
│       ├── specs/              # Design specifications
│       └── plans/              # Implementation plans
└── .env.example
```

## Quick Start

### Option 1: Demo (no setup required)

View pre-analyzed results instantly without backend or API key.

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.headless true
```

Then select **"Load saved result"** and choose `totalenergies_result.json`.

### Option 2: Full Pipeline

Run the complete analysis on any ESG report PDF.

**1. Backend setup:**

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export GROQ_API_KEY="your-groq-api-key"
uvicorn app:app --host 0.0.0.0 --port 8000
```

**2. Frontend setup (separate terminal):**

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.headless true
```

Then select **"Upload PDF"**, upload an ESG report, and click **"Analyze Report"**.

## Architecture

### Processing Pipeline

```
PDF Upload → Parse (PyMuPDF) → Index (RAG) → Extract Claims (ClimateBERT)
    → 7 Detection Modules (GPU models + LLM Judge)
    → Aggregation → Sin Scoring (LLM Scorer) → Report
```

### Stage 1: Detection Modules

Each extracted claim passes through all 7 modules. Each module uses a combination of GPU models and knowledge bases, then an LLM Judge (Groq) makes the final verdict.

| Module | GPU Models | Knowledge Base |
|--------|-----------|----------------|
| M1: Vagueness | ClimateBERT Specificity, Commitment, Zero-Shot | Vague Lexicon |
| M2: No Proof | ClimateBERT Fact-Check, TCFD, DeBERTa NLI | Proof Checklists |
| M3: Irrelevance | Zero-Shot Classifier | Irrelevance KB |
| M4: Lesser of Two Evils | FinBERT ESG, ClimateBERT Sentiment & Stance | Industry Risk |
| M5: Hidden Tradeoffs | FinBERT ESG Topic, TCFD, DeBERTa NLI | RAG Retriever |
| M6: False Labels | Zero-Shot Classifier | Legitimate Labels |
| M7: Fibbing | ClimateBERT Fact-Check, DeBERTa NLI, Sentiment, TAPAS | RAG Retriever |

### Stage 2: Sin Scoring

Each claim is scored across all 7 sins using weighted signals (0 / 0.5 / 1 scale). The LLM rates each signal; Python computes the weighted score.

```
Sin Score = 100 x sum(signal_weight x signal_value)
```

### Frontend Modes

| Mode | Description | Requires Backend |
|------|-------------|-----------------|
| Upload PDF | Full analysis pipeline | Yes |
| Load saved result | Display pre-analyzed results | No |
| Demo (mock data) | Layout preview with placeholder data | No |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Upload PDF, run full analysis |
| `/api/progress` | GET | Poll analysis progress (step/total) |
| `/api/report/{id}` | GET | Get full analysis report |
| `/api/report/{id}/summary` | GET | Get report summary |
| `/api/reports` | GET | List all completed reports |
| `/health` | GET | Health check |

## Requirements

- Python 3.10+
- CUDA GPU (recommended for GPU inference, falls back to CPU)
- Groq API key (free at https://console.groq.com)

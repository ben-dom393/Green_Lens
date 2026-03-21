# Green Lens

Greenwashing Detection Platform for ESG Reports. Analyzes environmental claims using NLP models, RAG evidence retrieval, and LLM-as-Judge architecture.

## Project Structure

```
Green_Lens/
├── backend/                    # FastAPI backend (main application)
│   ├── app.py                  # API server (POST /api/analyze, etc.)
│   ├── config.py               # Model IDs, thresholds, paths
│   ├── requirements.txt        # Python dependencies
│   ├── pipeline/
│   │   ├── claim_extractor.py  # ClimateBERT claim extraction + spaCy NER
│   │   ├── pdf_parser.py       # PyMuPDF PDF parsing (text + tables)
│   │   ├── llm/
│   │   │   ├── client.py       # Ollama/Groq LLM client
│   │   │   ├── judge.py        # LLM Judge (verdict decisions)
│   │   │   └── explainer.py    # (deprecated) template explanations
│   │   ├── modules/
│   │   │   ├── base.py         # Verdict dataclass + BaseModule ABC
│   │   │   ├── aggregator.py   # Combines all module verdicts into report
│   │   │   ├── vague_claims.py       # M1: Vague Claims detection
│   │   │   ├── no_proof.py           # M2: No Proof detection
│   │   │   ├── irrelevant_claims.py  # M3: Irrelevant Claims detection
│   │   │   ├── lesser_evil.py        # M4: Lesser of Two Evils
│   │   │   ├── hidden_tradeoffs.py   # M5: Hidden Tradeoffs
│   │   │   ├── fake_labels.py        # M6: Fake Labels
│   │   │   └── fibbing.py            # M7: Fibbing (false claims)
│   │   └── rag/
│   │       ├── indexer.py            # Document indexing (ChromaDB + BM25S)
│   │       ├── retriever.py          # Hybrid retrieval + cross-encoder reranking
│   │       ├── regulatory_indexer.py # Regulatory PDF indexing
│   │       └── regulatory_retriever.py
│   ├── data/
│   │   ├── vague_lexicon.json        # 261 vague terms (M1)
│   │   ├── proof_checklists.json     # 10 claim types, 47 evidence fields (M2)
│   │   ├── irrelevance_kb.json       # 15 irrelevance patterns (M3)
│   │   ├── industry_risk.json        # 13 sector risk profiles (M4, M5)
│   │   ├── legitimate_labels.json    # Eco-certification registry (M6)
│   │   └── external/
│   │       ├── regulatory/           # 10 regulatory PDFs (RAG indexed)
│   │       ├── datasets/             # Training datasets (ClimateBERT, etc.)
│   │       └── cases/                # TerraChoice case studies
│   └── tests/
├── ESG report/                 # 72 ESG report PDFs for testing
├── docs/
│   ├── reference/              # Architecture notes, research
│   └── superpowers/
│       ├── specs/              # Design specifications
│       └── plans/              # Implementation plans
└── .env.example
```

## Quick Start

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API

- `POST /api/analyze` — Upload PDF, run full greenwashing analysis
- `POST /api/analyze/text` — Analyze raw text
- `GET /api/report/{id}` — Get analysis report
- `GET /api/reports` — List all reports

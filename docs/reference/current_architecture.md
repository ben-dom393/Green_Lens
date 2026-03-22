# Green Lens — Current Architecture (v3: Full-Stack Connected)

> Updated 2026-03-22. This supersedes v2 (LLM Judge + 7 Modules, 2026-03-21).
> Changes: Frontend-backend integration, async pipeline, logging, sin scoring, progress bar.

---

## System Architecture

```
ESG Report (PDF)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  PDF Parser (PDFParser)                                  │
│  ├─ PyMuPDF (fitz) with sort=True for reading order      │
│  ├─ Column detection algorithm (multi-column layouts)    │
│  ├─ Header/footer filtering                              │
│  ├─ Table extraction (fitz find_tables)                  │
│  ├─ Section path tracking (hierarchical headings)        │
│  └─ element_role classification (context/claim/evidence) │
└──────┬───────────────────────────────────────────────────┘
       │  list[dict] → converted to list[DocumentElement] in app.py
       │  Fields: element_id, text, page, element_type, section_path,
       │          table_data, element_role
       ▼
┌──────────────────────────────────────────────────────────┐
│  Claim Extractor (ClaimExtractor)                        │
│  ├─ Filter: skip Title elements, skip text < 30 chars    │
│  ├─ spaCy sentencizer splits paragraphs → sentences      │
│  │  (with sentence_offset tracking)                      │
│  ├─ ClimateBERT environmental-claims classifier          │
│  │  (confidence threshold filters non-environmental)     │
│  ├─ spaCy NER + regex for entity/quantity extraction     │
│  ├─ Artifact soft-tagging (_detect_artifacts)            │
│  ├─ Per-sentence → Claim objects                         │
│  └─ _group_claims(): merge same-paragraph Claims →       │
│     ClaimGroup (return_groups=True in app.py)            │
│     - Merges entities, quantities, artifact_signals      │
│     - Picks representative_sentence (highest confidence) │
│     - ClaimGroup duck-types as Claim for M1-M7           │
└──────┬───────────────────────────────────────────────────┘
       │  list[ClaimGroup] (grouped by source paragraph)
       │  NVIDIA: 35 sentences → 21 groups
       │  Apple:  661 sentences → 164 groups
       ▼
┌──────────────────────────────────────────────────────────┐
│  BERTopic (document-level topic modeling)                │
│  → Identifies covered/missing topics for Hidden Tradeoffs│
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│       Detection Modules (sequential, all 7 active)       │
│                                                          │
│  1. Vague Claims     (specificity + commitment +         │
│                       DeBERTa zero-shot + lexicon +      │
│                       KeyBERT + LLM Judge)               │
│  2. No Proof         (fact-check NLI + DeBERTa NLI +     │
│                       TCFD + semantic checklist +         │
│                       regulatory RAG + LLM Judge)        │
│  3. Irrelevant       (DeBERTa zero-shot +                │
│                       regex KB + regulatory RAG +        │
│                       LLM Judge)                         │
│  4. Lesser Evil      (ESG-BERT + sentiment + stance +    │
│                       industry risk KB + LLM Judge)      │
│  5. Hidden Tradeoffs (BERTopic + TCFD + KeyBERT +        │
│                       ESG-BERT + materiality gap +       │
│                       LLM Judge)                         │
│  6. Fake Labels      (NER + cert KB + fuzzy match +      │
│                       DeBERTa zero-shot + LLM Judge)     │
│  7. Fibbing          (DeBERTa NLI contradiction +        │
│                       ClimateBERT fact-check +            │
│                       sentiment consistency +            │
│                       superlative detection + LLM Judge) │
└──────┬───────────────────────────────────────────────────┘
       │  list[Verdict] per module (with judgment dict from LLM)
       ▼
┌──────────────┐
│  Aggregator  │  → FinalReport with CategorySummary, risk_heatmap
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2: LLM Scorer (per-sin 0-100 scoring)           │
│  ├─ 7 sin-specific prompts, signals rated 0/0.5/1      │
│  ├─ Materiality multiplier: Low=1.0, Med=1.5, High=2.0 │
│  └─ Risk bands: Low(0-29), Moderate(30-59),            │
│     High(60-79), Critical(80+)                          │
└──────┬───────────────────────────────────────────────────┘
       │  scoring_summary with average_sin_scores (0-100)
       ▼
┌──────────────┐     ┌─────────────────────────────────────┐
│ FastAPI API  │ ──▶ │  Streamlit Frontend (port 8501)     │
│ (port 8000)  │     │  ├─ convert_backend_response()      │
│              │ ◀── │  │  sin_scores / 20 → 0-5 scale     │
│ GET /progress│     │  ├─ Radar chart + evidence cards     │
└──────────────┘     │  ├─ Progress bar (polls /progress)  │
                     │  └─ Demo mode (mock data fallback)   │
                     └─────────────────────────────────────┘
```

---

## Models Currently Loaded

| Component | Model ID | Size | Purpose |
|-----------|----------|------|---------|
| Claim Detection | `climatebert/environmental-claims` | ~82M | Classify sentences as environmental claims |
| Specificity | `climatebert/distilroberta-base-climate-specificity` | ~82M | Score claim vagueness |
| Commitment | `climatebert/distilroberta-base-climate-commitment` | ~82M | Action vs commitment |
| Zero-shot (shared) | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | ~184M | Fallback classification for M1, M3, M6 |
| Fact-Check | `amandakonet/climatebert-fact-checking` | ~82M | NLI: SUPPORTS/REFUTES/NOT_ENOUGH_INFO |
| NLI Contradiction | `cross-encoder/nli-deberta-v3-base` | ~184M | Contradiction detection for M2, M7 |
| TCFD Classifier | `climatebert/distilroberta-base-climate-tcfd` | ~82M | TCFD pillar classification for M2, M5 |
| ESG Sector | `yiyanghkust/finbert-esg-9-categories` | ~110M | Sector classification for M4, M5 |
| Climate Sentiment | `climatebert/distilroberta-base-climate-sentiment` | ~82M | Sentiment analysis for M4, M7 |
| Climate Stance | Custom stance model (if available) | ~82M | Stance detection for M4 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | ~22M | RAG + semantic matching |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~22M | Cross-encoder reranking |
| KeyBERT | `KeyBERT` with all-MiniLM-L6-v2 | ~22M | Key phrase extraction for M1, M5 |
| BERTopic | `BERTopic` with all-MiniLM-L6-v2 | ~22M | Document topic modeling |
| NER | `spaCy en_core_web_sm` | ~12M | Entity/quantity extraction + sentencizer |
| **LLM Judge** | **Groq `llama-3.3-70b-versatile`** | 70B (API) | Central verdict override + explanations |

---

## LLM Configuration (updated 2026-03-21)

**Groq only** — `llama-3.3-70b-versatile`, 70B model, free API tier.
- Rate limit: ~30 req/min, 14,400 req/day
- Retry logic: up to 5 retries with exponential backoff (30-120s) on 429/503 errors
- Parses `retry-after` and `x-ratelimit-reset-*` headers for precise wait times
- Requires `GROQ_API_KEY` environment variable

No local LLM fallback. Groq outperforms local models significantly on verdict quality.

The LLM Judge reviews every claim after NLP model signals are gathered. It receives:
- Claim text + section context
- All model signals (specificity score, commitment label, etc.)
- RAG evidence (truncated to 1000 chars)
- Regulatory evidence (truncated to 500 chars)
- Artifact signals (if any)

Output is a structured JSON with verdict, confidence, explanation, and evidence — stored for future UI use.

---

## Knowledge Bases

| KB File | Entries | Used by |
|---------|---------|---------|
| `data/vague_lexicon.json` | 261 terms | M1 (Vague Claims) |
| `data/proof_checklists.json` | 10 types, 47 fields | M2 (No Proof) |
| `data/irrelevance_kb.json` | 15 entries | M3 (Irrelevant Claims) |
| `data/industry_risk.json` | 15 sectors | M4 (Lesser Evil), M5 (Hidden Tradeoffs) |
| `data/label_registry.json` | 94 certifications | M6 (Fake Labels) |
| `data/external/` regulatory PDFs | 10 docs, 6268 chunks | All modules via Regulatory RAG |

---

## RAG Architecture

### In-Document RAG (per report)
- **Indexer:** ChromaDB (vectors) + BM25S (sparse) — built from the uploaded ESG report
- **Retriever:** Hybrid search with cross-encoder reranking (top-5)
- **Used by:** M2 (No Proof), M5 (Hidden Tradeoffs), M7 (Fibbing)

### Regulatory RAG (pre-built)
- **Source:** 10 regulatory PDFs (FTC Green Guides, GHG Protocol, SBTi, ISSB S2, etc.)
- **Indexer:** Separate ChromaDB collection, 6268 chunks
- **Used by:** M2, M3, M5, M6 — provides regulatory context for LLM Judge

---

## API Endpoints

```
POST /api/analyze         — Upload PDF, full analysis (runs in asyncio.to_thread)
POST /api/analyze/text    — Submit raw text for analysis
GET  /api/progress        — Poll analysis progress (step, total, percent, message)
GET  /api/report/{id}     — Full report with all verdicts + LLM judgments
GET  /api/report/{id}/category/{name} — Verdicts for one category
GET  /api/report/{id}/summary — Risk heatmap + verification tasks
GET  /api/reports         — List all analyzed reports
GET  /health              — Server health check
```

### Async Pipeline

The `/api/analyze` endpoint uses `asyncio.to_thread()` to run the pipeline synchronously
in a worker thread. This keeps the FastAPI event loop free so `/api/progress` can respond
during analysis. Without this, the server blocks all requests while processing a report.

### Logging

Rotating log files in `backend/logs/`:
- `green_lens.log` — all INFO+ messages (10MB max, 5 backups)
- `errors.log` — ERROR-only (5MB max, 3 backups)

All pipeline steps use `logger.info/warning/error` with `exc_info=True` for stack traces.
Per-module try/except ensures one failing module doesn't crash the whole pipeline.

### Frontend Integration

Score conversion: `scoring_summary.average_sin_scores[sin] / 20` (0-100 → 0-5).
Three naming systems bridged via dicts in `frontend/src/config.py`:
- `SIN_TO_METRIC`: backend sin names → frontend metric names
- `CATEGORY_TO_METRIC`: backend category names → frontend metric names

---

## Data Flow for Verdict Storage (for future UI)

Each verdict contains:
```python
Verdict(
    item_id="uuid",
    module_name="vague_claims",
    claim_id="claim_uuid",
    verdict="flagged|needs_verification|pass",
    explanation="NLP-generated explanation",
    missing_info=["scope boundary", "base year"],
    evidence=["evidence text from RAG"],
    page=12,
    claim_text="We are committed to sustainability...",
    section_path="Energy > Emissions",
    judgment={                          # ← LLM Judge output (stored for UI)
        "verdict": "flagged",
        "confidence": 0.85,
        "explanation": "This claim lacks...",
        "evidence_used": "The report does not...",
        "recommended_action": "Request specific..."
    }
)
```

The `judgment` dict is the key asset for future interactive UI — it contains the LLM's reasoning that can power:
- Paragraph highlighting with hover explanations
- "Why was this flagged?" drill-down
- Evidence trail linking claims to supporting/contradicting text

---

## File Structure

```
Green_Lens/
├── backend/
│   ├── app.py                    # FastAPI server (async pipeline + progress)
│   ├── config.py                 # Model IDs, thresholds, paths
│   ├── .env                      # GROQ_API_KEY (gitignored)
│   ├── requirements.txt          # Python dependencies
│   ├── logs/                     # Rotating log files (gitignored)
│   │   ├── green_lens.log        # All INFO+ messages (10MB, 5 backups)
│   │   └── errors.log            # ERROR-only (5MB, 3 backups)
│   ├── pipeline/
│   │   ├── pdf_parser.py         # PyMuPDF PDF parsing
│   │   ├── claim_extractor.py    # ClimateBERT + ClaimGroup + artifacts
│   │   ├── modules/
│   │   │   ├── base.py           # Verdict dataclass + BaseModule ABC
│   │   │   ├── aggregator.py     # FinalReport + CategorySummary
│   │   │   ├── vague_claims.py   # M1: Vague Claims
│   │   │   ├── no_proof.py       # M2: No Proof
│   │   │   ├── irrelevant_claims.py  # M3: Irrelevant Claims
│   │   │   ├── lesser_evil.py    # M4: Lesser of Two Evils
│   │   │   ├── hidden_tradeoffs.py   # M5: Hidden Tradeoffs
│   │   │   ├── fake_labels.py    # M6: Fake Labels
│   │   │   └── fibbing.py        # M7: Fibbing
│   │   ├── llm/
│   │   │   ├── client.py         # Groq API client (rate-limit retry)
│   │   │   ├── judge.py          # Stage 1: LLM Judge (7 prompt templates)
│   │   │   ├── scorer.py         # Stage 2: LLM Scorer (per-sin 0-100)
│   │   │   └── explainer.py      # (deprecated, replaced by judge.py)
│   │   └── rag/
│   │       ├── indexer.py        # ChromaDB + BM25S indexing
│   │       ├── retriever.py      # Hybrid search + reranking
│   │       ├── regulatory_indexer.py   # Regulatory PDF indexing
│   │       └── regulatory_retriever.py # Regulatory retrieval
│   ├── data/
│   │   ├── vague_lexicon.json
│   │   ├── proof_checklists.json
│   │   ├── irrelevance_kb.json
│   │   ├── industry_risk.json
│   │   ├── label_registry.json
│   │   └── external/             # Regulatory PDFs, datasets
│   └── test_results/             # Test output JSONs
├── frontend/
│   ├── app.py                    # Streamlit main (upload + progress bar)
│   ├── requirements.txt          # streamlit, plotly, requests
│   └── src/
│       ├── analysis.py           # run_analysis() + convert_backend_response()
│       ├── config.py             # API_URL, SIN_TO_METRIC, CATEGORY_TO_METRIC
│       ├── charts.py             # Plotly radar chart
│       ├── ui_components.py      # Render functions (header, cards, evidence)
│       └── sample_data.py        # Mock data for demo mode
├── docs/
│   └── reference/
│       ├── current_architecture.md   # ← THIS FILE (current state)
│       ├── mvp_architecture.md       # Original MVP plan (historical)
│       ├── model_structure.md        # Original Chinese design doc (historical)
│       ├── knowledge_bases.md        # KB specification
│       └── research_notes.md         # Research findings
├── ESG report/                   # 72+ test ESG report PDFs
└── .gitignore                    # Excludes .env, logs/, ESG reports
```

---

## Test Results (2026-03-21)

| Report | Elements | Claims | Modules | Time | Status |
|--------|----------|--------|---------|------|--------|
| NVIDIA | 168 | 35 | All 7 + LLM | ~914s | Pass |
| Chevron | 1200+ | 200+ | All 7 + LLM | ~2298s | Pass |
| Coca-Cola | 300+ | 100+ | All 7 + LLM | ~433s | Pass |

### Chunking Comparison (4 reports)

| Report | PDF Elements | Sentence Claims | ClaimGroups | Reduction | Merged | Single |
|--------|-------------|----------------|-------------|-----------|--------|--------|
| NVIDIA | 168 | 35 | 21 | 40% | 9 | 12 |
| Apple | 564 | 661 | 164 | 75% | 113 | 51 |
| Bank of America | 488 | 315 | 105 | 67% | 62 | 43 |
| Honeywell | 522 | 172 | 82 | 52% | 31 | 51 |

Note: ClaimGroups are what actually get processed by M1-M7. The grouping merges
same-paragraph sentence claims into single analysis units, significantly reducing
LLM API calls (e.g., Apple: 661 LLM calls → 164 with grouping).

---

## Known Limitations & Future Improvements

1. **Environmental-only detection:** The claim extractor uses `climatebert/environmental-claims`
   which only detects environmental/climate claims. Social claims (diversity, labor, human rights)
   and governance claims (board composition, ethics, cybersecurity) are not reliably detected.
   Expanding to S and G would require additional models (e.g., `finbert-esg-9-categories` as a
   pre-filter, or separate social-washing and governance-washing detection pipelines).

2. **Cross-column PDF merging:** Some multi-column PDF pages produce merged text from different
   columns (e.g., NVIDIA p13, p18). The column detection algorithm handles most cases but not all.

3. **Navigation/sidebar text in claims:** Some claims contain appended navigation text
   (e.g., "Responsible Business") from PDF sidebar elements. Artifact soft-tagging flags these
   but does not remove them (zero-deletion principle).

4. **Claim count scaling:** Dense environmental reports (e.g., Apple 661 sentences) produce many
   claims. ClaimGroup aggregation reduces this significantly (661 → 164) but very large reports
   may still be slow with LLM Judge enabled.

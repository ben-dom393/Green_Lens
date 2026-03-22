# Green Lens — Research Notes & Technical Decisions

> Summary of research findings that informed the MVP architecture.
> This is a reference document for the team, not user-facing.

---

## NLP Models Selected (with rationale)

### Claim Extraction: ClimateBERT Environmental Claims

- **Model:** `climatebert/environmental-claims` (HuggingFace)
- **Architecture:** DistilRoBERTa, pre-trained on 2M+ climate paragraphs
- **Task:** Binary classification — "is this sentence an environmental claim?"
- **Training data:** Expert-annotated sentences from corporate annual reports, sustainability reports, earnings transcripts (2,647 sentences by 16 domain experts)
- **Why chosen:** Directly solves our first pipeline step. No fine-tuning needed. Runs locally.
- **Companion dataset:** `climatebert/environmental_claims` on HuggingFace (MIT license)

### Vagueness Detection: ClimateBERT Specificity (NEW — key discovery)

- **Model:** `climatebert/distilroberta-base-climate-specificity` (HuggingFace)
- **Task:** Binary classification — "is this climate claim specific or vague?"
- **Why this is critical:** This model was trained EXACTLY for our Module 1 task. It replaces the need for a custom vagueness classifier. Combined with our rule-based lexicon, this gives us a very strong Module 1.
- **Companion dataset:** `climatebert/climate_specificity` on HuggingFace

### Fact-Checking: ClimateBERT Fact-Checking (NEW — key discovery)

- **Model:** `amandakonet/climatebert-fact-checking` (HuggingFace)
- **Task:** Entailment — given (claim, evidence), predict SUPPORTS / REFUTES / NOT_ENOUGH_INFO
- **Training data:** CLIMATE-FEVER dataset (1,535 real-world claims, 7,675 evidence pairs)
- **Why this is critical:** Directly solves Module 2's core task — checking whether evidence in the report supports a given claim.

### Full ClimateBERT Family (all ~82M params, all run on CPU)

| Model | Task | Use in our pipeline |
|-------|------|---------------------|
| `climatebert/environmental-claims` | Is this an environmental claim? | Claim Extractor |
| `climatebert/distilroberta-base-climate-specificity` | Is this specific or vague? | Module 1 |
| `climatebert/distilroberta-base-climate-sentiment` | Sentiment toward climate topics | Module 5 (hidden tradeoffs) |
| `climatebert/distilroberta-base-climate-commitment` | Is this about commitments/actions? | Claim type enrichment |
| `climatebert/distilroberta-base-climate-tcfd` | TCFD disclosure category | Structural analysis |
| `climatebert/distilroberta-base-climate-detector` | Is this paragraph climate-related? | Topic gate |
| `amandakonet/climatebert-fact-checking` | Does evidence support claim? | Module 2 |

### ESG Topic Filtering: ESG-BERT or FinBERT-ESG

- **Best option:** `nbroad/ESG-BERT` — F1=0.90, 26 ESG categories (vs 0.79 for vanilla BERT)
- **Alternative:** `yiyanghkust/finbert-esg-9-categories` — 9 categories, ~14,000 training sentences
- **Simpler option:** `yiyanghkust/finbert-esg` — 4 categories (E/S/G/None)
- **Additional:** `ESGBERT/EnvironmentalBERT-environmental` — binary environmental detection, 2K samples
- **Why chosen:** Topic gate to filter to environmental paragraphs before claim extraction.

### Greenwashing Category Classification: SetFit

- **Framework:** SetFit (HuggingFace)
- **Base model:** `all-MiniLM-L6-v2` (~80MB)
- **Why chosen:** With only 8-16 examples per category, SetFit outperforms GPT-3 and vanilla fine-tuned transformers. Training takes ~30 seconds on GPU, few minutes on CPU. Model is tiny (~80MB).
- **How it works:** Contrastive learning on sentence pairs → lightweight classification head on embeddings.
- **Fallback:** `facebook/bart-large-mnli` for zero-shot classification if we can't collect enough labeled examples.

### Embeddings for RAG: all-MiniLM-L6-v2

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Size:** ~80MB
- **Speed:** Fastest option (~14.7ms per 1K tokens)
- **Why chosen:** Best speed/quality tradeoff for hackathon. Runs easily on CPU. The accuracy gap vs. larger models (BGE, E5) matters less when our document corpus is small (single ESG report).
- **Better alternatives if needed:** `BAAI/bge-base-en-v1.5` (768d, ~440MB, better accuracy) or `intfloat/e5-base-v2`.

### Reranker: Cross-Encoder MiniLM

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Task:** Score (query, document) relevance on 0-1 scale
- **Why chosen:** Standard free reranker. Dramatically improves retrieval quality. Retrieve top-20 with BM25/vector, rerank, take top-5.
- **Usage:** `sentence-transformers` library → `CrossEncoder` class.

### Local LLM: Ollama

- **Tool:** Ollama
- **Recommended models by VRAM:**
  - **8GB VRAM:** `qwen3:8b` (Q4_K_M, ~6GB) — best overall at 8GB, strong structured JSON output, 32K context
  - **16GB VRAM:** `qwen3:14b` (Q4_K_M, ~10GB) — better instruction quality; or `phi-4` (14B) for reasoning
  - **Low-end:** `llama3.2:3b` (~2GB) — fast fallback; or `phi-4-mini` (3.8B, ~3.5GB)
- **Why Qwen3 over Llama/Mistral:** Proven structured JSON output support with Ollama, strongest instruction following at this size
- **Why Ollama:** Simplest setup (`ollama pull qwen3:8b`), native JSON output, REST API at `localhost:11434`, Python library, LangChain integration
- **Use cases:** Explanation generation, claim structuring, reasoning over evidence, structured output for aggregation

---

## Vague Language Detection: Research Findings

### Academic Background
- **CoNLL-2010 Shared Task** on hedge detection — benchmark dataset, best F1: 67.5% (Wikipedia), 86.2% (biomedical)
- **"Finding Hedges by Chasing Weasels" (ACL 2009)** — used Wikipedia weasel-word tags as training signal

### Our Approach: Specificity Score (rule-based, no ML needed)

Combine multiple signals into a composite score:

1. **Hedge phrase density** — count hedge/weasel phrases per sentence (from lexicon)
2. **Numeric specificity** — presence of concrete numbers, dates, percentages, units
3. **Named entity density** — presence of specific organizations, standards, certifications
4. **Temporal specificity** — concrete dates vs. vague timeframes
5. **Commitment verb strength** — "will reduce by 50% by 2030" vs. "aims to explore"
6. **Methodology reference** — mentions LCA, GHG Protocol, ISO, third-party audit, etc.

This can be implemented with spaCy NER + regex + lexicon. No ML model required.

### Key Research Papers on Greenwashing NLP

| Paper | Year | Key contribution |
|-------|------|-----------------|
| "Corporate Greenwashing Detection in Text — a Survey" | 2025 (arXiv 2502.07541) | Comprehensive survey of 61 works; identifies achievable vs. hard sub-tasks |
| "Leveraging Language Models to Detect Greenwashing" | 2023 (arXiv 2311.01469) | Contrastive + ordinal ranking for graded greenwashing detection |
| "Detecting Greenwashing Hints in ESG Reports" | 2025 (SwissText/ACL) | Practical detection methods for ESG reports specifically |
| "ESG-washing detection in corporate sustainability reports" | 2024 (ScienceDirect) | Systematic approach to ESG report analysis |
| "EmeraldMind: Knowledge Graph–Augmented Framework" | 2025 (arXiv 2512.11506) | KG-augmented detection — relevant for our KB approach |

### Key Finding from Survey (2502.07541)
> "Climate topic identification, climate risk classification, and characterization of environmental claim types now achieve near-perfect performance with fine-tuned models, and even simple keyword-based approaches achieve surprisingly strong results. However, tasks requiring nuance, subjectivity, or reasoning — such as distinguishing specific from vague commitments or assessing rhetorical deception — remain challenging."

**Implication for us:** Our modules 1-4 (rule-based + small models) should work well. Modules 5-7 (requiring reasoning) are genuinely harder — this validates our decision to defer scoring and use conservative `needs_verification` outputs.

---

## RAG Architecture Decisions

### Vector Store: ChromaDB
- **Why:** "SQLite for embeddings" — simple, persistent, metadata filtering, 5-10 lines to set up
- **When to use FAISS instead:** Only for millions of vectors with GPU — not our case

### Sparse Search: BM25S
- **Package:** `pip install bm25s`
- **Why over rank_bm25:** Orders of magnitude faster (scipy sparse matrices). Drop-in replacement.
- **When to use:** Keyword-heavy queries (certification names, "Scope 1", specific regulation names)

### Hybrid Search Strategy
- Retrieve top-20 from both BM25S and ChromaDB
- Combine using **Reciprocal Rank Fusion (RRF):** `score = 1/(k + rank)` per retriever, then sum
- Rerank top-20 combined results with cross-encoder → take top-5
- This avoids the incompatible score scale problem between dense and sparse retrieval

---

## Pre-Download Data Strategy (vs. LLM)

### Decision: Pre-download everything that's static

| Data category | Pre-download | Live query |
|--------------|:---:|:---:|
| Regulatory principles | Yes | No |
| Certification registries | Yes | No |
| Banned substance lists | Yes | No |
| Industry risk profiles | Yes | No |
| Greenwashing case examples | Yes | No |
| Company-specific verification | No | Future enhancement |
| Real-time news/events | No | Future enhancement |
| Current certification validity | No | Future enhancement |

### Rationale
- Deterministic, reproducible results
- Zero API cost at runtime
- No external dependency failures
- Faster processing
- Easier to test and debug
- Students can review and curate the data

---

## Datasets Available on HuggingFace

| Dataset | HuggingFace ID | Content | Use for |
|---------|---------------|---------|---------|
| ClimateBERT Environmental Claims | `climatebert/environmental_claims` | Annotated sentences from corporate reports | Evaluation + few-shot examples |
| ClimaBench | `iceberg-nlp/climabench` | Multi-task climate NLP benchmark | Evaluation of claim detection |
| Climate Policy Radar docs | `ClimatePolicyRadar/all-document-text-data` | Climate policy document corpus | Reference for regulatory principles |
| ClimDetect | `ClimDetect/ClimDetect` | Climate change detection benchmark | Reference |
| Climate Evaluation | `eci-io/climate-evaluation` | Climate evaluation dataset | Reference |

---

## Free LLM API Options (researched)

| Provider | Free tier details | Best for |
|----------|------------------|----------|
| **Groq** | Free tier with Llama 3.x, Mixtral. ~30 req/min. Very fast inference. | Explanation generation when local LLM is slow |
| **HuggingFace Inference API** | Free for most models. Rate limited. | Running ClimateBERT, FinBERT-ESG without local GPU |
| **Google Gemini** | Free tier: Gemini Flash, ~15 req/min. | Complex multi-step reasoning (if needed) |
| **Ollama (local)** | No limits, runs on your hardware | Primary inference for MVP |

**MVP strategy:** Use Ollama as primary. Fall back to Groq free tier if local hardware is insufficient for 7B models.

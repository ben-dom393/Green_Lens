# Green Lens: LLM Judge Architecture + Modules 5-7 + All-Module Enhancements

**Date:** 2026-03-21
**Status:** Design — awaiting approval

---

## 1. Overview

Upgrade all 7 greenwashing detection modules from rule/model-based verdict systems to an **LLM-as-Judge** architecture. Each module's existing models and knowledge bases become **signal providers** — their outputs are packaged into structured evidence that an LLM judge uses to make the final verdict.

Additionally, build three new modules (Hidden Tradeoffs, Fake Labels, Fibbing) and enhance all existing modules with specialized NLP models best-fit for each detection task.

### Design Principles

- **Models decide, KBs provide factual knowledge** — but now the LLM is the decision-maker, and all other models are evidence gatherers
- **Graceful degradation** — if LLM is unavailable, modules fall back to their existing model-based verdict logic
- **Structured output for UI** — every LLM judgment stores reasoning, highlight spans, suggestions, and evidence links for future interactive frontend
- **GPU available** — use larger, more accurate model variants (DeBERTa-v3-large over base)
- **Specialized models per module** — no constraint to reuse the same model everywhere

---

## 2. LLM Judge Core (`pipeline/llm/judge.py`)

### New Component: `LLMJudge`

Uses the existing `LLMClient` (Ollama → Groq fallback). Provides a single entry point for all modules.

```python
class LLMJudge:
    def judge_claim(
        self,
        module_name: str,        # e.g. "vague_claims"
        claim_text: str,
        signals: dict,           # model outputs specific to this module
        evidence: list[dict],    # RAG-retrieved evidence passages
        kb_context: dict,        # relevant KB data (lexicon, checklists, etc.)
    ) -> JudgmentResult
```

### `JudgmentResult` dataclass

```python
@dataclass
class JudgmentResult:
    verdict: str              # "flagged" | "pass" | "needs_verification"
    confidence: float         # 0.0-1.0
    reasoning: str            # Human-readable explanation (for UI display)
    highlight_spans: list[str]  # Exact text spans to highlight in report viewer
    suggestion: str           # What would make this claim acceptable
    evidence_used: list[str]  # Which evidence passages influenced decision
    severity: str             # "high" | "medium" | "low"
    raw_llm_response: str     # Full LLM output for debugging/storage
```

### Prompt Templates

Each module has its own prompt template stored as a string constant in `judge.py`. Templates follow this structure:

```
SYSTEM: You are a greenwashing analyst specializing in [module-specific domain].

USER:
CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
{module-specific signals formatted as key-value pairs}

EVIDENCE FROM DOCUMENT:
{RAG evidence passages, numbered}

REGULATORY CONTEXT:
{regulatory evidence if available}

KB CONTEXT:
{relevant knowledge base entries}

Respond ONLY with valid JSON:
{
  "verdict": "flagged" | "pass" | "needs_verification",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 sentence explanation>",
  "highlight_spans": ["<exact text spans from the claim to highlight>"],
  "suggestion": "<what would make this claim acceptable, or empty if pass>",
  "evidence_used": ["<which evidence passages influenced your decision>"],
  "severity": "high" | "medium" | "low"
}
```

### JSON Parsing

The judge must parse LLM output as JSON. Strategy:
1. Try `json.loads(response)` directly
2. If fails, extract JSON block from markdown code fences (```json ... ```)
3. If fails, regex extract `{...}` block
4. If all fail, return a fallback JudgmentResult with verdict from the module's model-based logic

### Integration with Existing Modules

Each module's `analyze()` method changes from:

```
(before) signals → verdict logic → Verdict
(after)  signals → LLMJudge.judge_claim() → JudgmentResult → Verdict
```

If LLM is unavailable, the existing verdict logic runs as fallback.

---

## 3. Enhanced Verdict Storage

### Changes to `Verdict` dataclass (`base.py`)

Add one new optional field with `field(default=None)`:

```python
@dataclass
class Verdict:
    item_id: str
    module_name: str
    claim_id: str
    verdict: str
    explanation: str
    missing_info: list[str]
    evidence: list[dict]
    page: int
    claim_text: str
    section_path: list[str]
    judgment: dict | None = field(default=None)  # NEW: Full LLM JudgmentResult
```

The `judgment` dict contains the complete `JudgmentResult` — verdict, confidence, reasoning, highlight_spans, suggestion, evidence_used, severity, and module_signals.

### `Verdict.create()` factory update

Add `judgment` as an explicit optional parameter:

```python
@staticmethod
def create(
    module_name: str,
    claim_id: str,
    verdict: str,
    explanation: str,
    page: int,
    claim_text: str,
    section_path: list[str] | None = None,
    missing_info: list[str] | None = None,
    evidence: list[dict] | None = None,
    judgment: dict | None = None,      # NEW
) -> "Verdict":
    # If LLM judgment provided, use its reasoning as explanation
    if judgment and "reasoning" in judgment:
        explanation = judgment["reasoning"]
    return Verdict(
        item_id=str(uuid.uuid4()),
        module_name=module_name,
        claim_id=claim_id,
        verdict=verdict,
        explanation=explanation,
        missing_info=missing_info or [],
        evidence=evidence or [],
        page=page,
        claim_text=claim_text,
        section_path=section_path or [],
        judgment=judgment,
    )
```

### Aggregator update (`aggregator.py`)

The aggregator builds item dicts manually at lines 91-101 and must include the new `judgment` field:

```python
# In aggregator.py, update the item dict building:
item = {
    "item_id": v.item_id,
    "verdict": v.verdict,
    "claim_text": v.claim_text,
    "explanation": v.explanation,
    "missing_info": v.missing_info,
    "evidence": v.evidence,
    "page": v.page,
    "section_path": v.section_path,
    "judgment": v.judgment,       # NEW: include full LLM judgment
}
```

### API Response

No API endpoint changes needed. The `judgment` field flows through the aggregator into the JSON response. Frontend can access:
- `verdict.judgment.reasoning` for "why was this flagged?"
- `verdict.judgment.highlight_spans` for text highlighting
- `verdict.judgment.suggestion` for improvement tooltips
- `verdict.judgment.severity` for color coding
- `verdict.judgment.confidence` for confidence bars

### Explainer Deprecation

The existing `pipeline/llm/explainer.py` is **superseded** by `LLMJudge`. During the upgrade:
1. Modules that currently import `Explainer` (M1 vague_claims, M2 no_proof) should remove those imports
2. The `explainer.py` file is kept but no longer called — it can be deleted after all modules are migrated
3. All LLM-generated explanations now come from `JudgmentResult.reasoning`

---

## 4. New Models Added to Config

### `config.py` additions

```python
# --- Commitment model (move from hardcoded string to config) ---
COMMITMENT_MODEL = "climatebert/distilroberta-base-climate-commitment"

# --- Zero-shot classification (upgrade from BART) ---
ZEROSHOT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

# --- Contradiction / NLI (new) ---
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# --- TCFD classification (new) ---
TCFD_MODEL = "climatebert/distilroberta-base-climate-tcfd"

# --- Climate sentiment (new) ---
CLIMATE_SENTIMENT_MODEL = "climatebert/distilroberta-base-climate-sentiment"

# --- Climate stance detection (new) ---
CLIMATE_STANCE_MODEL = "rldekkers/bert-base-uncased-finetuned-climate-stance-detection"

# --- Table fact verification (new) ---
TABLE_FACT_MODEL = "google/tapas-base-finetuned-tabfact"
```

### Total Model Inventory

| Model ID | ~Size | GPU | Modules |
|---|---|---|---|
| `climatebert/environmental-claims` | 82M | Yes | Claim Extractor |
| `climatebert/distilroberta-base-climate-specificity` | 82M | Yes | M1 |
| `climatebert/distilroberta-base-climate-commitment` | 82M | Yes | M1 |
| `amandakonet/climatebert-fact-checking` | 82M | Yes | M2, M7 |
| `yiyanghkust/finbert-esg-9-categories` | 110M | Yes | M4, M5 |
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | Yes | M2, M5, M6 (embeddings) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | Yes | RAG reranking |
| `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | 870M | Yes | M1, M3, M6 |
| `cross-encoder/nli-deberta-v3-base` | 350M | Yes | M2, M7 |
| `climatebert/distilroberta-base-climate-tcfd` | 82M | Yes | M2, M5 |
| `climatebert/distilroberta-base-climate-sentiment` | 82M | Yes | M4, M7 |
| `rldekkers/bert-base-uncased-finetuned-climate-stance-detection` | 440M | Yes | M4 |
| `google/tapas-base-finetuned-tabfact` | 440M | Yes | M7 |
| **BERTopic** (library, uses existing embeddings) | — | — | M5 |
| **KeyBERT** (library, uses existing embeddings) | — | — | M1, M5 |
| `qwen3:8b` / `llama-3.3-70b-versatile` | 8B/70B | Ollama/API | All modules (LLM Judge) |

---

## 5. Module-by-Module Design

### Module 1: Vague Claims (`vague_claims.py`)

**Signals gathered (existing + new):**

| Signal | Model/Tool | Output |
|---|---|---|
| Specificity score | `climatebert/distilroberta-base-climate-specificity` | float 0-1 |
| Commitment type | `climatebert/distilroberta-base-climate-commitment` | "action" or "commitment" |
| Zero-shot vagueness | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | scores for ["specific measurable claim", "vague aspirational claim", "general statement"] |
| Vague term count | Vague lexicon (`vague_lexicon.json`) | count + matched terms |
| Positive signals | Regex (numbers, dates, standards, scope) | count + matched patterns |
| Key phrases | **KeyBERT** | top 3-5 key phrases from the claim |

**Changes:**
1. Replace `facebook/bart-large-mnli` with `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
2. Add KeyBERT key phrase extraction
3. Package all signals → LLM Judge
4. Keep existing verdict logic as fallback

**LLM Judge prompt focus:** "Is this claim specific enough to be verifiable, or is it vague greenwashing?"

---

### Module 2: No Proof (`no_proof.py`)

**Signals gathered (existing + new):**

| Signal | Model/Tool | Output |
|---|---|---|
| Evidence passages | RAG HybridRetriever | top-k text passages + scores |
| Regulatory evidence | Regulatory RAG | regulatory text + source citations |
| ClimateBERT fact-check | `amandakonet/climatebert-fact-checking` | SUPPORTS/REFUTES/NEI per passage |
| DeBERTa NLI | `cross-encoder/nli-deberta-v3-base` | entailment/contradiction/neutral scores per passage |
| Proof checklist coverage | semantic matching + `proof_checklists.json` | missing fields list |
| TCFD coverage | `climatebert/distilroberta-base-climate-tcfd` | TCFD category of claim (Governance/Strategy/Risk/Metrics) |

**Changes:**
1. Add `cross-encoder/nli-deberta-v3-base` as second NLI signal alongside ClimateBERT
2. Add TCFD classification to assess disclosure quality context
3. Both NLI results (ClimateBERT domain-specific + DeBERTa general) sent to LLM Judge
4. Package all signals → LLM Judge
5. Keep existing verdict logic as fallback

**LLM Judge prompt focus:** "Does the evidence adequately support this claim? What proof is missing?"

---

### Module 3: Irrelevant Claims (`irrelevant_claims.py`) — Major Upgrade

**Signals gathered (new architecture):**

| Signal | Model/Tool | Output |
|---|---|---|
| Regex KB match | `irrelevance_kb.json` patterns | matched pattern + regulation (fast path) |
| Zero-shot classification | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | scores for ["legally mandated requirement", "banned substance compliance", "industry standard practice", "voluntary environmental action"] |
| Regulatory cross-check | Regulatory RAG | search for the claimed attribute in regulatory docs |

**Changes:**
1. Keep regex KB as **fast path** — if a known pattern matches, it's a high-confidence flag
2. Add zero-shot classification for claims that DON'T match regex — catches novel irrelevance patterns
3. Add Regulatory RAG search — find whether the claimed attribute is required by law
4. Package all signals → LLM Judge
5. Fallback: regex-only verdict (current behavior)

**LLM Judge prompt focus:** "Is this environmental claim actually just compliance with existing law, making it irrelevant as a voluntary green effort?"

---

### Module 4: Lesser of Two Evils (`lesser_evil.py`)

**Signals gathered (existing + new):**

| Signal | Model/Tool | Output |
|---|---|---|
| Sector classification | `yiyanghkust/finbert-esg-9-categories` | ESG category → sector mapping |
| Industry risk profile | `industry_risk.json` | risk level, primary impacts, material topics |
| Green buzzword detection | Regex | buzzword matches |
| Climate sentiment | `climatebert/distilroberta-base-climate-sentiment` | positive/negative/neutral sentiment score |
| Climate stance | `rldekkers/bert-base-uncased-finetuned-climate-stance-detection` | stance toward climate topic |
| Specificity check | Regex (numbers, methodology) | has quantitative backing? |

**Changes:**
1. Add climate sentiment model — detect overly positive framing in high-impact sectors
2. Add climate stance detection — defensive stance + green claims = stronger red flag
3. Package all signals → LLM Judge
4. Fallback: existing ESG-BERT + keyword verdict logic

**LLM Judge prompt focus:** "Is this green claim proportional to the company's actual environmental impact, or is it distracting from larger harms?"

---

### Module 5: Hidden Tradeoffs (`hidden_tradeoffs.py`) — NEW

**Sin:** Highlighting one narrow green attribute while ignoring larger environmental impacts.

**Signals gathered:**

| Signal | Model/Tool | Output |
|---|---|---|
| Sector identification | `yiyanghkust/finbert-esg-9-categories` (reuse from M4) | sector + risk profile |
| Expected material topics | `industry_risk.json` → `expected_material_topics` | list of topics this sector should address |
| Document topic coverage | **BERTopic** with `all-MiniLM-L6-v2` | discovered topics in the full report |
| Topic gap search | **RAG retriever** | for each expected material topic, search if it appears in the document |
| TCFD coverage | `climatebert/distilroberta-base-climate-tcfd` | which TCFD pillars are covered vs missing |
| Key themes of claim | **KeyBERT** | what narrow theme does this specific claim focus on |
| Scope narrowing detection | Regex | patterns like "our offices", "one facility", "packaging only", "in [region]" |

**Architecture:**
1. ESG-BERT identifies sector from document context (can reuse M4's result — cache at pipeline level)
2. Look up expected material topics from `industry_risk.json`
3. Run BERTopic on all document elements to discover what topics the report covers
4. For each expected material topic NOT discovered by BERTopic, run RAG search to confirm absence
5. Run TCFD classifier on the claim to understand its disclosure category
6. KeyBERT extracts what the claim actually focuses on
7. Regex detects scope-narrowing language
8. LLM Judge receives: the claim, its narrow focus, what topics the sector should address, which are missing from the report, TCFD coverage gaps

**LLM Judge prompt focus:** "Is this claim cherry-picking a minor green attribute while the company's major environmental impacts go unaddressed in this report?"

**Fallback (no LLM):** Flag if >=50% of expected material topics are absent from the document AND the claim focuses on a non-material topic.

---

### Module 6: Fake Labels (`fake_labels.py`) — NEW

**Sin:** Using self-created, fake, or misleading eco-certifications.

**Signals gathered:**

| Signal | Model/Tool | Output |
|---|---|---|
| Certification extraction | **Regex patterns** | extract "X Certified", "X Approved", "X Standard", "X Label", trademark symbols |
| Organization NER | **spaCy NER** (ORG entities) | extract organization names near certification language |
| Legitimate label lookup | **New KB: `legitimate_labels.json`** | exact match against ~100+ known certifications |
| Fuzzy label matching | **Sentence-transformers** cosine similarity | compare extracted name against all legitimate label names — catch near-misses |
| Zero-shot classification | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | classify as "third-party verified certification" vs "self-awarded label" vs "industry standard" vs "marketing language" |
| Regulatory label check | **Regulatory RAG** | search regulatory docs for the certification name |

**New KB: `legitimate_labels.json`**

Structure:
```json
{
  "version": "1.0.0",
  "labels": [
    {
      "name": "Forest Stewardship Council (FSC)",
      "short_names": ["FSC", "FSC Certified", "FSC Mix"],
      "issuing_body": "Forest Stewardship Council",
      "category": "forestry",
      "url": "https://fsc.org",
      "what_it_certifies": "Responsible forest management and chain of custody"
    },
    {
      "name": "Energy Star",
      "short_names": ["Energy Star", "ENERGY STAR"],
      "issuing_body": "U.S. Environmental Protection Agency",
      "category": "energy_efficiency",
      ...
    }
  ]
}
```

~100+ entries covering: FSC, PEFC, Energy Star, EU Ecolabel, Blue Angel, Nordic Swan, Cradle to Cradle, B Corp, ISO 14001, ISO 14064, ISO 50001, LEED, BREEAM, Fair Trade, Rainforest Alliance, UTZ, Marine Stewardship Council, Responsible Jewellery Council, Global Organic Textile Standard, OEKO-TEX, Green Seal, EcoLogo, SFI, Carbon Trust, Science Based Targets, CDP, GRI, TCFD, SASB, etc.

**Architecture:**
1. Regex extracts certification-like phrases from claim text
2. spaCy NER extracts ORG entities near certification language
3. Exact match against `legitimate_labels.json` — if found, claim references a real label → pass
4. If not found, fuzzy match via sentence-transformers — if close but not exact, flag as suspicious
5. Zero-shot classifies the claim's certification language
6. Regulatory RAG checks if any regulatory doc references the certification
7. LLM Judge evaluates all signals

**LLM Judge prompt focus:** "Is the certification or eco-label referenced in this claim legitimate, self-created, or misleadingly similar to a real certification?"

**Fallback (no LLM):** Flag if extracted certification name is not found in legitimate_labels.json AND fuzzy similarity to any legitimate label is < 0.8.

---

### Module 7: Fibbing (`fibbing.py`) — NEW

**Sin:** Outright false or fabricated environmental claims that can be checked against available evidence.

**Signals gathered:**

| Signal | Model/Tool | Output |
|---|---|---|
| Superlative/absolute detection | **Regex** | "first to", "only company", "100%", "zero emissions", "completely eliminated", "entirely" |
| ClimateBERT fact-check | `amandakonet/climatebert-fact-checking` | SUPPORTS/REFUTES/NEI for evidence pairs |
| DeBERTa contradiction detection | `cross-encoder/nli-deberta-v3-base` | contradiction/entailment/neutral scores |
| Internal consistency search | **RAG retriever** (targeted) | search for same topic elsewhere in report — find contradicting passages |
| Climate sentiment consistency | `climatebert/distilroberta-base-climate-sentiment` | sentiment of claim vs sentiment of surrounding evidence |
| Table fact verification | `google/tapas-base-finetuned-tabfact` | verify claim against extracted tables (SUPPORTED/REFUTED) |
| Quantity cross-check | Claim extractor quantities | extracted numbers from claim + evidence passages |

**Architecture:**
1. Regex detects superlative and absolute language in the claim
2. RAG retrieves passages about the SAME topic from elsewhere in the document
3. ClimateBERT fact-checker evaluates each (claim, evidence) pair
4. DeBERTa NLI cross-encoder evaluates each (claim, evidence) pair for contradiction
5. If tables were extracted from the PDF, TAPAS verifies the claim against table data
6. Climate sentiment model checks for sentiment mismatches between claim and evidence
7. Quantity extraction compares numbers mentioned in claim vs evidence
8. LLM Judge receives all signals

**Key distinction from Module 2:** Module 2 asks "is there evidence?" Module 7 asks "is the claim actually false?" Module 2 flags absence of proof. Module 7 flags presence of contradiction.

**LLM Judge prompt focus:** "Based on the evidence within this document, is this claim factually false, internally contradicted, or implausibly absolute?"

**Fallback (no LLM):** Flag if DeBERTa NLI finds contradiction score > 0.7 for any evidence pair, OR if TAPAS returns REFUTED, OR if superlative claim has no supporting evidence.

---

## 6. Aggregator MODULE_ORDER Update

The current aggregator has `"needs_verification_claims"` as the 7th slot. This is replaced with `"fibbing"`:

```python
MODULE_ORDER = [
    ("vague_claims", "Vague Claims"),
    ("no_proof", "No Proof"),
    ("irrelevant_claims", "Irrelevant Claims"),
    ("lesser_of_two_evils", "Lesser of Two Evils"),
    ("hidden_tradeoffs", "Hidden Tradeoffs"),
    ("fake_labels", "Fake Labels"),
    ("fibbing", "Fibbing"),                        # CHANGED from needs_verification_claims
]
```

The old `"needs_verification_claims"` category is removed. The "needs_verification" verdict status still exists as a verdict value within any module — it is not a standalone module.

---

## 7. New Libraries Required

```
pip install bertopic keybert
```

BERTopic and KeyBERT both use the existing `sentence-transformers/all-MiniLM-L6-v2` model for embeddings — no additional model downloads needed for these libraries.

spaCy is already installed (used by ClaimExtractor). Module 6 reuses the existing `en_core_web_sm` model for ORG entity extraction. No additional spaCy downloads needed.

TAPAS requires:
```
pip install torch-scatter  # optional, for TAPAS table processing
```

---

## 8. Pipeline Integration (`app.py`)

### New lazy-loaded components

```python
def get_hidden_tradeoffs_module():
    from pipeline.modules.hidden_tradeoffs import HiddenTradeoffsModule
    return HiddenTradeoffsModule()

def get_fake_labels_module():
    from pipeline.modules.fake_labels import FakeLabelsModule
    return FakeLabelsModule()

def get_fibbing_module():
    from pipeline.modules.fibbing import FibbingModule
    return FibbingModule()

def get_llm_judge():
    from pipeline.llm.judge import LLMJudge
    return LLMJudge()
```

### Analyze endpoint changes

After Module 4, add:

```python
# Module 5: Hidden Tradeoffs
hidden_mod = get_hidden_tradeoffs_module()
hidden_verdicts = hidden_mod.analyze(claims, retriever=retriever, ...)
all_verdicts.extend(hidden_verdicts)

# Module 6: Fake Labels
fake_mod = get_fake_labels_module()
fake_verdicts = fake_mod.analyze(claims)
all_verdicts.extend(fake_verdicts)

# Module 7: Fibbing
fib_mod = get_fibbing_module()
fib_verdicts = fib_mod.analyze(claims, retriever=retriever, ...)
all_verdicts.extend(fib_verdicts)
```

### Shared model caching via `**kwargs`

Several models are reused across modules. To avoid loading the same model twice, shared instances are passed via `**kwargs` in `analyze()` calls from `app.py`:

- `llm_judge=judge` — single `LLMJudge` instance passed to all modules
- `nli_model=nli` — DeBERTa NLI cross-encoder shared between M2 and M7
- `zeroshot_model=zs` — DeBERTa zero-shot shared between M1, M3, M6
- `sentiment_model=sent` — climate sentiment shared between M4 and M7
- `sector_result=sector` — M4's sector classification result cached and passed to M5
- `document_topics=topics` — BERTopic result (document-level) computed once and passed to M5

Each module lazily accesses these from `kwargs.get()` and loads its own instance as fallback if not provided.

### Table data flow for TAPAS (M7)

The PDF parser already extracts table data (returned as `table_data` in element dicts). Currently `app.py` discards this when converting to `DocumentElement`. To support M7:

1. Store raw table data from `simple_text_extract()` in a separate list
2. Pass to M7 via `kwargs`: `table_data=tables`
3. M7 feeds tables to TAPAS for claim verification

### BERTopic document-level analysis

BERTopic runs ONCE on all `DocumentElement.text` values (not claim texts) to discover report-level topic coverage. This happens in `app.py` before module analysis:

1. Collect all element texts: `all_texts = [e.text for e in elements if len(e.text) > 50]`
2. Run BERTopic: `topic_model.fit_transform(all_texts)`
3. Extract discovered topics as list of keyword sets
4. Pass to M5 via `kwargs`: `document_topics=discovered_topics`

### GPU Memory Budget

Expected concurrent model loading (~4-5 GB VRAM total):
- ClimateBERT family (5 models × ~330MB each) ≈ 1.6 GB
- DeBERTa zero-shot large ≈ 1.7 GB
- DeBERTa NLI base ≈ 0.7 GB
- ESG-BERT + stance ≈ 1.1 GB
- TAPAS ≈ 0.9 GB
- Sentence-transformers + reranker ≈ 0.2 GB

Total: ~6.2 GB VRAM. Models are lazy-loaded, so peak usage depends on which modules are active. A GPU with 8+ GB VRAM handles this comfortably.

### `legitimate_labels.json` creation

This KB will be created as part of Phase 3 implementation. Sources:
- Ecolabel Index (ecolabelindex.com) — 456 global eco-labels
- ISEAL Alliance members
- ISO environmental standards (14001, 14064, 50001)
- Major certification bodies (FSC, MSC, Rainforest Alliance, Fair Trade, etc.)

The file is committed alongside code in `backend/data/`.

### `/api/analyze/text` endpoint

Both `/api/analyze` (PDF) and `/api/analyze/text` (raw text) endpoints run all modules. The new M5/M6/M7 modules and LLM Judge kwargs must be added to BOTH endpoints. The text endpoint follows the same pattern as the PDF endpoint.

---

## 9. Execution Flow (Updated)

```
1. PDF Upload
2. PDF Parsing → DocumentElement list (with tables extracted)
3. RAG Indexing → ChromaDB + BM25S
4. Regulatory RAG Indexing (if not already done)
5. Claim Extraction → ClimateBERT environmental-claims
6. BERTopic → document-level topic analysis (for M5)
7. Module 1: Vague Claims → signals → LLM Judge → Verdict
8. Module 2: No Proof → signals → LLM Judge → Verdict
9. Module 3: Irrelevant Claims → signals → LLM Judge → Verdict
10. Module 4: Lesser Evil → signals → LLM Judge → Verdict (cache sector result)
11. Module 5: Hidden Tradeoffs → signals + topic gaps → LLM Judge → Verdict
12. Module 6: Fake Labels → signals → LLM Judge → Verdict
13. Module 7: Fibbing → signals + contradiction search → LLM Judge → Verdict
14. Aggregation → FinalReport with all 7 categories
```

---

## 10. Testing Strategy

### Phase 1: Core infrastructure
- Build and test `LLMJudge` with one module (Module 1 — Vague Claims)
- Verify JSON parsing, fallback logic, JudgmentResult structure
- Test with Ollama and Groq

### Phase 2: Upgrade existing modules (M1-M4)
- Integrate LLM Judge into each existing module one at a time
- Add new models (DeBERTa zero-shot, NLI, sentiment, stance, TCFD)
- Test each module individually

### Phase 3: Build new modules (M5-M7)
- Build M5 (Hidden Tradeoffs) with BERTopic + topic gap analysis
- Build M6 (Fake Labels) with legitimate_labels.json + fuzzy matching
- Build M7 (Fibbing) with DeBERTa NLI + TAPAS

### Phase 4: End-to-end test
- Run full pipeline on NVIDIA ESG report
- Manually review all 7 module outputs
- Verify LLM judgment quality and structured output

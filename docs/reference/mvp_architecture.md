# Green Lens — MVP Architecture (Case B: Local LLM + Free API + NLP)

> **SUPERSEDED** by `current_architecture.md` as of 2026-03-21.
> This document is kept as historical reference for the original MVP plan.
> Key changes since this doc: Ollama removed (Groq-only), all 7 modules built,
> LLM Judge architecture added, ClaimGroup aggregation added.

---

## Core Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | English only | Simplifies lexicons, models, and evaluation |
| Deployment | Case B only (local LLM + free API + NLP) | MVP first, no paid API dependency |
| User input | ESG report PDF only | No optional documents; reference data is pre-built |
| Temporal scope | Single-year analysis | Flag unverifiable claims; verify against pre-built DB if data exists |
| "Outright Lies" | Renamed to "Needs Verification" | Reduces legal risk; outputs `needs_verification` not `flagged` |
| Scoring | Deferred | Output = opinion text + flagged sentence/paragraph; scoring policy designed later |
| User feedback loop | Deferred | Not MVP priority |
| Image/visual analysis | Deferred | Not MVP priority |
| PDF text extraction | Teammate handles | Focus on detection pipeline, not parsing |
| PDF table extraction | Deferred | Text-first; tables attempted only after text extraction is stable |

---

## System Architecture Overview

```
ESG Report (PDF)
       │
       ▼
┌──────────────┐
│  PDF Parser  │  ← Teammate handles (unstructured + PyMuPDF)
│  (text only) │
└──────┬───────┘
       │  document_elements.jsonl
       ▼
┌──────────────────┐
│  Claim Extractor │  ← ClimateBERT + spaCy rules + keyword fallback
└──────┬───────────┘
       │  claims.json
       ▼
┌──────────────────────────────────────────┐
│       Detection Modules (sequential)      │
│                                           │
│  1. Vague Claims      (lexicon + rules)   │
│  2. No Proof           (in-doc RAG)       │
│  3. Irrelevant Claims  (KB lookup)        │
│  4. Lesser of Two Evils (sector + rules)  │
│  5. Hidden Tradeoffs   (cross-section)    │
│  6. Fake Labels        (registry lookup)  │
│  7. Needs Verification (consistency)      │
└──────┬───────────────────────────────────┘
       │  verdicts per module
       ▼
┌──────────────┐
│  Aggregator  │  → final_report.json
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌────────────┐
│ FastAPI API  │ ──▶ │  Frontend  │
└──────────────┘     └────────────┘
```

---

## Technical Stack

### Core NLP Models (all free, run locally)

| Component | Model / Tool | Purpose | Size |
|-----------|-------------|---------|------|
| Claim detection | `climatebert/environmental-claims` | Binary: is this an environmental claim? | ~82M params |
| Vagueness detection | `climatebert/distilroberta-base-climate-specificity` | Binary: is claim specific or vague? | ~82M params |
| Fact-checking | `amandakonet/climatebert-fact-checking` | Entailment: does evidence support claim? | ~82M params |
| ESG topic gate | `nbroad/ESG-BERT` (F1=0.90) or `finbert-esg-9-categories` | 26/9 ESG categories | ~110M params |
| Greenwashing classification | SetFit with `all-MiniLM-L6-v2` | Few-shot into 7 categories (8-16 examples each) | ~80MB |
| Zero-shot fallback | `facebook/bart-large-mnli` | When no labeled examples available | ~1.5GB |
| Embeddings | `all-MiniLM-L6-v2` | Vector index for in-document RAG | ~80MB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Rerank evidence relevance | ~80MB |
| NER + rules | spaCy `en_core_web_sm` | Entity extraction, dependency parsing | ~12MB |
| Local LLM | Ollama (`qwen3:8b` or `qwen3:14b`) | Reasoning, explanation generation | 6-10GB |

### Infrastructure

| Component | Tool | Why |
|-----------|------|-----|
| Vector store | ChromaDB | Simple, persistent, metadata filtering, "SQLite for embeddings" |
| Sparse search | BM25S (`bm25s` package) | Fast BM25, hybrid search with vectors |
| Backend | FastAPI | Already set up on backend branch |
| Local LLM serving | Ollama (`qwen3:8b`) | Simplest local LLM setup, native JSON output, REST API |
| Data storage | JSON/SQLite | Simple, no database server needed |

### Free LLM API Options (for when local LLM is insufficient)

| Provider | Free tier | Best model | Rate limit | Use case |
|----------|-----------|------------|------------|----------|
| Groq | Free, no credit card | LLaMA 3.3 70B | ~30 RPM, 6K TPM | Fast inference, JSON mode, high quality |
| Google Gemini | Free, no credit card | Gemini 2.5 Flash | 10 RPM, 250 RPD, 1M context | Analyze entire ESG reports in one pass |
| HuggingFace Inference API | Free | ClimateBERT, FinBERT-ESG | ~few hundred RPH | Run specialized models without local GPU |
| OpenRouter | Free, no credit card | DeepSeek V3/R1 | ~20 RPM | Aggregator — one API key, many models |
| Cerebras | Free, no credit card | LLaMA variants | 30 RPM, 1M tok/day | Extremely fast inference |

---

## Pre-Built Knowledge Bases

> **Key decision:** Users only upload ESG reports. All reference data (regulations, certifications, known patterns) is pre-built into the system.

### 1. `vague_lexicon.json` — Vague/hedge language patterns

Used by: **Module 1 (Vague Claims)**

Categories:
- **Weasel quantifiers:** some, many, most, several, numerous, various, certain
- **Vague commitments:** strive to, aim to, aspire to, endeavor to, seek to, work towards, committed to exploring
- **Peacock terms:** industry-leading, world-class, best-in-class, significant, substantial, meaningful
- **Greenwashing buzzwords (without metrics):** sustainable, green, eco-friendly, clean, responsible, environmentally conscious
- **Modal hedges:** could, might, may, possibly, potentially, perhaps
- **Vague time references:** soon, in the near future, in due course, over time, going forward
- **Passive attribution:** it is believed, it is said, reportedly

### 2. `regulatory_principles.json` — Key regulatory requirements

Used by: **Modules 2, 3, 4, 5**

Sources to include:
- FTC Green Guides (US) — rules on environmental marketing claims
- UK CMA Green Claims Code — 6 principles for green claims
- EU Green Claims Directive — requirements for substantiation
- ISSB/IFRS S2 — climate disclosure requirements (Scope 1/2/3, methodology, boundaries)
- GHG Protocol — corporate accounting standard for emissions
- ASA (UK Advertising Standards Authority) — guidance on environmental claims

### 3. `label_registry.json` — Known certifications and ecolabels

Used by: **Module 6 (Fake Labels)**

Structure per entry:
```json
{
  "label_name": "FSC Certified",
  "official_org": "Forest Stewardship Council",
  "iso_type": "Type I (ISO 14024)",
  "verification": "Third-party audited",
  "lookup_url": "https://info.fsc.org/certificate.php",
  "common_misuses": ["Using FSC logo without valid certificate number"]
}
```

Target: 100-300 common certifications (ISO 14001, FSC, LEED, B Corp, Fair Trade, EU Ecolabel, Energy Star, etc.)

### 4. `irrelevance_kb.json` — Claims with no informational value

Used by: **Module 3 (Irrelevant Claims)**

Structure per entry:
```json
{
  "pattern": "CFC-free",
  "reason": "CFCs banned globally under Montreal Protocol since 1987",
  "regulation": "Montreal Protocol",
  "applies_to": "all products"
}
```

Examples: CFC-free, lead-free paint (banned), asbestos-free (banned), compliant with [universally mandatory regulation]

### 5. `industry_risk.json` — High-impact sector classification

Used by: **Module 4 (Lesser of Two Evils)**

Structure:
```json
{
  "sector": "fossil_fuels",
  "keywords": ["oil", "gas", "petroleum", "coal", "refining"],
  "risk_level": "very_high",
  "primary_impacts": ["Scope 1 emissions", "Scope 3 downstream combustion"],
  "note": "Single-point environmental improvements do not offset core business impact"
}
```

Sectors: fossil fuels, tobacco, fast fashion, cement, mining, aviation, agrochemicals, arms

### 6. `greenwashing_cases.json` — Known cases as few-shot examples

Used by: **All modules (for SetFit training + LLM few-shot prompts + evaluation)**

Sources:
- TerraChoice "Sins of Greenwashing" studies (original case examples)
- EU Commission sweep study results
- UK ASA rulings on environmental claims
- Academic papers with annotated examples
- ClimateBERT environmental claims dataset (`climatebert/environmental_claims` on HuggingFace)

---

## Module Implementation Details

### Module 1: Vague Claims (EASIEST)

**Input:** Extracted claims
**Technique:** ClimateBERT specificity model + rule-based hybrid
**External data needed:** `vague_lexicon.json` only

Pipeline:
1. Run `climatebert/distilroberta-base-climate-specificity` on each claim → binary: specific or vague
2. For claims flagged as vague, score on a **specificity checklist:**
   - Has concrete numbers/percentages? (+)
   - Has specific dates/deadlines? (+)
   - Has methodology reference (LCA, GHG Protocol, etc.)? (+)
   - Has named certifications/standards? (+)
   - Has boundary/scope definition? (+)
   - Contains vague buzzwords from lexicon? (-)
   - Contains hedge/weasel words? (-)
3. Combine model output + checklist into vagueness assessment
4. Output: flagged sentence + missing specifics list

**Key model:** ClimateBERT specificity was trained specifically on corporate climate text to distinguish vague from specific claims — this is exactly our task.

### Module 2: No Proof (EASY-MEDIUM)

**Input:** Extracted claims
**Technique:** In-document RAG (hybrid search) + ClimateBERT fact-checking
**External data needed:** `regulatory_principles.json`, `proof_checklists.json`

Pipeline:
1. For each claim, search the SAME ESG report for supporting evidence
2. Use hybrid search: BM25S (keyword) + ChromaDB (semantic) + cross-encoder reranker
3. Run `amandakonet/climatebert-fact-checking` on (claim, evidence) pairs → SUPPORTS / REFUTES / NOT_ENOUGH_INFO
4. Check against a **proof checklist** from `proof_checklists.json` (adapted per claim type):
   - Emissions claim → needs Scope, boundary, base year, methodology, third-party verification
   - Recycling claim → needs percentage, methodology, certification
   - Target/commitment → needs timeline, baseline, interim milestones
5. Output: claim + evidence found (or not) + entailment result + missing proof items

**Key model:** ClimateBERT fact-checking was trained on CLIMATE-FEVER (1,535 claims, 7,675 evidence pairs) for exactly this: checking whether evidence supports a climate claim.

### Module 3: Irrelevant Claims (MEDIUM)

**Input:** Extracted claims
**Technique:** KB lookup + pattern matching
**External data needed:** `irrelevance_kb.json`

Pipeline:
1. Normalize claim → extract (object, attribute, qualifier)
2. Match against `irrelevance_kb.json` entries
3. If match found → flag as irrelevant with reason from KB
4. If no match → pass (or flag as `needs_verification` if claim seems trivially true)

### Module 4: Lesser of Two Evils (MEDIUM)

**Input:** Extracted claims + company sector
**Technique:** Sector classification + rules
**External data needed:** `industry_risk.json`

Pipeline:
1. Extract company sector from ESG report (company profile, revenue breakdown)
2. Match against `industry_risk.json`
3. If high-risk sector + green claim detected → flag with explanation template
4. Output avoids moral judgment; states factual risk context

### Module 5: Hidden Tradeoffs (MEDIUM-HARD)

**Input:** All claims + full document
**Technique:** Cross-section analysis + topic distribution
**External data needed:** `regulatory_principles.json` (materiality expectations per sector)

Pipeline:
1. Map all claims by topic (embedding clustering or keyword grouping)
2. Identify "claim density" vs "data/evidence density" per topic
3. Flag areas where claims are high but data is low
4. Cross-reference with sector materiality: if biggest impact area (e.g., Scope 3 for manufacturing) has few claims or data → flag
5. Output: claim-level flags + cross-section gap report

### Module 6: Fake Labels (HARD)

**Input:** Extracted claims (specifically those mentioning certifications)
**Technique:** NER + registry lookup
**External data needed:** `label_registry.json`

Pipeline:
1. Extract certification mentions: NER + regex ("certified", "ISO", "FSC", "LEED", etc.)
2. Match against `label_registry.json`
3. If in registry → check if claim usage is consistent with certification scope
4. If NOT in registry → flag as `needs_verification`
5. Output: certification + verification status + registry reference

### Module 7: Needs Verification (HARDEST — formerly "Outright Lies")

**Input:** All extracted claims + data points
**Technique:** Internal consistency check + external DB check
**External data needed:** All KBs + any historical data available

Pipeline:
1. **Sub-module A: Internal consistency**
   - Extract same metric across different sections/tables
   - Check for contradictions (same year, same scope, different values)
2. **Sub-module B: Verifiable claims**
   - Extract "we achieved X certification" / "we reached Y target" type claims
   - Check against pre-built DB if data exists → verified / contradicted
   - If no data → output `needs_verification` with specific verification task
3. Output: always conservative — `verified`, `contradicted`, or `needs_verification`

---

## API Contract (FastAPI Endpoints)

```
POST /api/analyze
  Body: { file: PDF upload }
  Response: { report_id: string, status: "processing" }

GET /api/report/{report_id}
  Response: final_report.json (full structure from model_structure.md)

GET /api/report/{report_id}/category/{category_name}
  Response: { items: [...] } for a specific greenwashing category

GET /api/report/{report_id}/summary
  Response: { risk_heatmap: {...}, total_flags: int, verification_tasks: [...] }

GET /api/health
  Response: { status: "ok" }
```

---

## Output Schema

Same as defined in `model_structure.md` section "輸出資料結構", with these changes:

1. `outright_lies` category renamed to `needs_verification_claims`
2. All items in modules 6 and 7 default to `verdict: "needs_verification"` unless DB confirms/contradicts
3. `confidence` field deferred — not computed in MVP
4. `severity` field deferred — not computed in MVP
5. `explanation` field = natural language opinion from local LLM or template

---

## Ground Truth / Evaluation Strategy

Since we're students and can't guarantee correctness:

1. **Collect known greenwashing cases** as examples (not as training data):
   - `climatebert/environmental_claims` — 2,647 annotated sentences (HuggingFace, MIT license)
   - `climatebert/climate_specificity` — vague vs specific climate claims dataset
   - A3CG dataset — aspect-action classification for greenwashing (ACL 2025, GitHub: `keanepotato/a3cg_greenwash`)
   - DizzyPanda1 GreenwashingDetectionDataset — real company claims + accusations (GitHub)
   - CLIMATE-FEVER — 1,535 claims with 7,675 evidence pairs for fact verification (HuggingFace)
   - TerraChoice studies (original Seven Sins examples, PDFs available)
   - UK ASA rulings database (searchable at asa.org.uk/rulings)
   - FTC enforcement cases (~100 cases since 1991, ~36 since 2013)
   - EU Commission sweep study (344 claims from 27 countries)

2. **Use cases four ways:**
   - **Few-shot examples** in LLM prompts
   - **SetFit training** (8-16 examples per category)
   - **ClimateBERT fine-tuning data** (specificity + claims datasets)
   - **Sanity check evaluation** — run system on known cases, manually check if flags make sense

3. **Honest framing:** System identifies potential greenwashing *risks* and *areas needing verification*, not definitive judgments.

---

## Implementation Priority Order

| Phase | Module | Estimated Effort | Dependencies |
|-------|--------|-----------------|--------------|
| **Phase 1** | Claim Extractor (ClimateBERT + spaCy) | 1-2 days | PDF parser output |
| **Phase 2** | Module 1: Vague Claims | 0.5-1 day | Claim Extractor + vague_lexicon.json |
| **Phase 3** | Module 2: No Proof | 1-2 days | Claim Extractor + ChromaDB + BM25S |
| **Phase 4** | Aggregator + API + basic output | 1 day | Modules 1-2 working |
| **Phase 5** | Module 3: Irrelevant Claims | 0.5 day | irrelevance_kb.json |
| **Phase 6** | Module 4: Lesser of Two Evils | 0.5 day | industry_risk.json |
| **Phase 7** | Module 5: Hidden Tradeoffs | 1-2 days | Full document analysis |
| **Phase 8** | Module 6: Fake Labels | 1 day | label_registry.json |
| **Phase 9** | Module 7: Needs Verification | 1-2 days | All other modules |

**MVP milestone:** Phase 1-4 complete = working end-to-end demo with 2 modules.

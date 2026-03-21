# LLM Judge Architecture + Modules 5-7 + All-Module Enhancements

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade all 7 greenwashing detection modules to LLM-as-Judge architecture, build 3 new modules (Hidden Tradeoffs, Fake Labels, Fibbing), and enhance all modules with specialized NLP models.

**Architecture:** Each module gathers signals from specialized models/KBs, then sends a structured evidence packet to an LLM Judge (Ollama/Groq) that makes the final verdict. The LLM's full reasoning is stored in a `judgment` dict on the Verdict for future UI consumption. Graceful degradation to existing model-based logic when LLM is unavailable.

**Tech Stack:** FastAPI, PyTorch, transformers, sentence-transformers, BERTopic, KeyBERT, spaCy, ChromaDB, BM25S, Ollama (qwen3:8b), Groq (llama-3.3-70b)

**Spec:** `docs/superpowers/specs/2026-03-21-llm-judge-all-modules-design.md`

---

## File Structure

### New files to create:
- `backend/pipeline/llm/judge.py` — LLMJudge class with per-module prompt templates and JSON parsing
- `backend/pipeline/modules/hidden_tradeoffs.py` — Module 5: Hidden Tradeoffs detection
- `backend/pipeline/modules/fake_labels.py` — Module 6: Fake Labels detection
- `backend/pipeline/modules/fibbing.py` — Module 7: Fibbing (false claims) detection
- `backend/data/legitimate_labels.json` — Knowledge base of ~100+ real eco-certifications
- `backend/tests/test_judge.py` — Tests for LLMJudge
- `backend/tests/test_modules.py` — Integration tests for all 7 modules

### Files to modify:
- `backend/pipeline/modules/base.py` — Add `judgment` field to Verdict
- `backend/config.py` — Add new model constants
- `backend/pipeline/modules/aggregator.py` — Update MODULE_ORDER + include judgment in items
- `backend/pipeline/modules/vague_claims.py` — Integrate LLM Judge + replace BART with DeBERTa + add KeyBERT
- `backend/pipeline/modules/no_proof.py` — Integrate LLM Judge + add DeBERTa NLI + TCFD
- `backend/pipeline/modules/irrelevant_claims.py` — Add zero-shot + regulatory RAG + LLM Judge
- `backend/pipeline/modules/lesser_evil.py` — Add sentiment + stance + LLM Judge
- `backend/app.py` — Wire up M5-M7, shared models, BERTopic, table data flow
- `backend/requirements.txt` — Add bertopic, keybert

---

## Chunk 1: Core Infrastructure (Tasks 1-4)

### Task 1: Install new dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add new dependencies to requirements.txt**

Add these lines to `backend/requirements.txt`:

```
# Topic modeling & keyword extraction
bertopic
keybert

# Already installed via transformers, but noting new models:
# MoritzLaurer/deberta-v3-large-zeroshot-v2.0
# cross-encoder/nli-deberta-v3-base
# climatebert/distilroberta-base-climate-tcfd
# climatebert/distilroberta-base-climate-sentiment
# rldekkers/bert-base-uncased-finetuned-climate-stance-detection
# google/tapas-base-finetuned-tabfact
```

- [ ] **Step 2: Install dependencies**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && pip install bertopic keybert`

---

### Task 2: Add new model constants to config.py

**Files:**
- Modify: `backend/config.py`

- [ ] **Step 1: Add new model constants**

Add after the existing `RERANKER_MODEL` line in `backend/config.py`:

```python
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
```

- [ ] **Step 2: Verify config imports work**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "from config import ZEROSHOT_MODEL, NLI_MODEL, TCFD_MODEL, CLIMATE_SENTIMENT_MODEL, CLIMATE_STANCE_MODEL, TABLE_FACT_MODEL, COMMITMENT_MODEL; print('All config constants loaded OK')"`

Expected: `All config constants loaded OK`

---

### Task 3: Update Verdict dataclass and Aggregator

**Files:**
- Modify: `backend/pipeline/modules/base.py`
- Modify: `backend/pipeline/modules/aggregator.py`

- [ ] **Step 1: Add `judgment` field to Verdict dataclass**

In `backend/pipeline/modules/base.py`, add `from dataclasses import dataclass, field` is already imported. Add the new field after `section_path`:

```python
    section_path: list[str]
    judgment: dict | None = field(default=None)  # Full LLM JudgmentResult
```

- [ ] **Step 2: Update `Verdict.create()` factory method**

In `backend/pipeline/modules/base.py`, update the `create` method to accept `judgment`:

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
        judgment: dict | None = None,
    ) -> "Verdict":
        """Convenience factory that auto-generates an item_id."""
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

- [ ] **Step 3: Update Aggregator MODULE_ORDER**

In `backend/pipeline/modules/aggregator.py`, change the MODULE_ORDER:

```python
    MODULE_ORDER = [
        ("vague_claims", "Vague Claims"),
        ("no_proof", "No Proof"),
        ("irrelevant_claims", "Irrelevant Claims"),
        ("lesser_of_two_evils", "Lesser of Two Evils"),
        ("hidden_tradeoffs", "Hidden Tradeoffs"),
        ("fake_labels", "Fake Labels"),
        ("fibbing", "Fibbing"),
    ]
```

- [ ] **Step 4: Add `judgment` to aggregator item dict**

In `backend/pipeline/modules/aggregator.py`, update the item dict building (around line 92-101) to include judgment:

```python
                item = {
                    "item_id": v.item_id,
                    "verdict": v.verdict,
                    "claim_text": v.claim_text,
                    "explanation": v.explanation,
                    "missing_info": v.missing_info,
                    "evidence": v.evidence,
                    "page": v.page,
                    "section_path": v.section_path,
                    "judgment": v.judgment,
                }
```

- [ ] **Step 5: Verify Verdict + Aggregator still work**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
from pipeline.modules.base import Verdict
v = Verdict.create(module_name='test', claim_id='c1', verdict='flagged', explanation='test', page=1, claim_text='test claim', judgment={'reasoning': 'LLM says flagged', 'confidence': 0.9})
print(f'verdict={v.verdict}, explanation={v.explanation}, judgment={v.judgment}')
assert v.explanation == 'LLM says flagged', 'reasoning should override explanation'
assert v.judgment['confidence'] == 0.9
print('Verdict OK')
from pipeline.modules.aggregator import Aggregator
a = Aggregator()
report = a.aggregate([v], 1, 'test.pdf')
d = report.to_dict()
cat = d['categories'][0]  # test module won't match MODULE_ORDER, check no crash
print(f'Aggregator OK, categories={len(d[\"categories\"])}')
"`

Expected: Verdict OK, Aggregator OK with 7 categories

---

### Task 4: Build LLMJudge core

**Files:**
- Create: `backend/pipeline/llm/judge.py`

- [ ] **Step 1: Create judge.py with JudgmentResult and LLMJudge**

Create `backend/pipeline/llm/judge.py`:

```python
"""LLM Judge: makes final greenwashing verdicts using structured evidence from modules."""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class JudgmentResult:
    """Structured output from the LLM judge."""

    verdict: str = "needs_verification"
    confidence: float = 0.5
    reasoning: str = ""
    highlight_spans: list[str] = field(default_factory=list)
    suggestion: str = ""
    evidence_used: list[str] = field(default_factory=list)
    severity: str = "medium"
    raw_llm_response: str = ""
    module_signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# Per-module prompt templates
# ------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a greenwashing analyst. You evaluate environmental claims "
    "from corporate ESG reports for signs of greenwashing. "
    "Give concise, factual assessments based on the evidence provided. "
    "Always respond with valid JSON only — no markdown, no extra text."
)

PROMPT_TEMPLATES: dict[str, str] = {
    "vague_claims": """Evaluate whether this environmental claim is vague or specific.

CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
- Specificity score: {specificity_score} (0=vague, 1=specific)
- Commitment type: {commitment_label}
- Zero-shot result: {zeroshot_result}
- Vague terms found: {vague_terms}
- Positive signals (numbers/dates/standards): {positive_signals}
- Key phrases: {key_phrases}

EVIDENCE FROM DOCUMENT:
{evidence}

REGULATORY CONTEXT:
{regulatory}

Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",

    "no_proof": """Evaluate whether this environmental claim has adequate supporting evidence.

CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
- ClimateBERT fact-check: {fact_check_results}
- DeBERTa NLI: {nli_results}
- Proof checklist coverage: {checklist_coverage}
- Missing evidence fields: {missing_fields}
- TCFD category: {tcfd_category}
- Evidence passages found: {evidence_count}

EVIDENCE FROM DOCUMENT:
{evidence}

REGULATORY CONTEXT:
{regulatory}

Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",

    "irrelevant_claims": """Evaluate whether this environmental claim is actually just compliance with existing law (making it irrelevant as a voluntary green effort).

CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
- Regex KB match: {regex_match}
- Zero-shot classification: {zeroshot_result}

REGULATORY CONTEXT:
{regulatory}

Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",

    "lesser_of_two_evils": """Evaluate whether this green claim is proportional to the company's actual environmental impact, or is it distracting from larger harms (lesser of two evils).

CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
- Sector: {sector} (risk level: {risk_level})
- Primary industry impacts: {primary_impacts}
- Climate sentiment: {sentiment}
- Climate stance: {stance}
- Green buzzwords found: {buzzwords}
- Has quantitative backing: {has_numbers}

EVIDENCE FROM DOCUMENT:
{evidence}

Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",

    "hidden_tradeoffs": """Evaluate whether this claim cherry-picks a minor green attribute while the company's major environmental impacts go unaddressed.

CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
- Sector: {sector}
- Expected material topics for this sector: {expected_topics}
- Topics found in document: {found_topics}
- Missing material topics: {missing_topics}
- TCFD coverage: {tcfd_coverage}
- Claim focuses on: {claim_focus}
- Scope narrowing detected: {scope_narrowing}

EVIDENCE FROM DOCUMENT:
{evidence}

Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",

    "fake_labels": """Evaluate whether the certification or eco-label referenced in this claim is legitimate, self-created, or misleading.

CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
- Extracted certification names: {cert_names}
- Legitimate label match: {legitimate_match}
- Fuzzy similarity to known labels: {fuzzy_match}
- Zero-shot classification: {zeroshot_result}
- Organizations mentioned: {organizations}

REGULATORY CONTEXT:
{regulatory}

Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",

    "fibbing": """Evaluate whether this claim is factually false, internally contradicted, or implausibly absolute based on in-document evidence.

CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
- Superlatives/absolutes detected: {superlatives}
- ClimateBERT fact-check: {fact_check_results}
- DeBERTa contradiction scores: {contradiction_scores}
- Sentiment consistency: {sentiment_consistency}
- Quantities in claim: {claim_quantities}
- Table verification: {table_verification}

CONTRADICTING PASSAGES FROM SAME REPORT:
{contradicting_passages}

SUPPORTING EVIDENCE:
{evidence}

Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",
}


class LLMJudge:
    """Central LLM judge for all greenwashing detection modules."""

    def __init__(self) -> None:
        self._client = LLMClient()
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if any LLM backend is reachable."""
        if self._available is None:
            import os
            self._available = (
                self._client._check_ollama()
                or bool(os.environ.get("GROQ_API_KEY"))
            )
        return self._available

    def judge_claim(
        self,
        module_name: str,
        claim_text: str,
        signals: dict,
        evidence: list[dict] | None = None,
        kb_context: dict | None = None,
    ) -> JudgmentResult | None:
        """Send structured evidence to LLM and get a verdict.

        Returns None if LLM is unavailable (caller should use fallback logic).
        """
        if not self.is_available():
            return None

        template = PROMPT_TEMPLATES.get(module_name)
        if not template:
            logger.warning("No prompt template for module '%s'", module_name)
            return None

        # Build evidence text
        evidence_text = "None found."
        if evidence:
            parts = []
            for i, ev in enumerate(evidence[:5], 1):
                text = ev.get("text", str(ev))[:300]
                page = ev.get("page", "?")
                parts.append(f"[{i}] (p.{page}) {text}")
            evidence_text = "\n".join(parts)

        # Build regulatory text
        regulatory_text = "None available."
        if kb_context and kb_context.get("regulatory"):
            reg_parts = []
            for r in kb_context["regulatory"][:3]:
                doc = r.get("document_name", r.get("source", "unknown"))
                pg = r.get("page", "?")
                txt = r.get("text", "")[:200]
                reg_parts.append(f"[{doc}, p.{pg}] {txt}")
            regulatory_text = "\n".join(reg_parts)

        # Merge all template variables
        template_vars = {
            "claim_text": claim_text,
            "page": signals.get("page", "?"),
            "section_path": " > ".join(signals.get("section_path", [])) or "N/A",
            "evidence": evidence_text,
            "regulatory": regulatory_text,
        }
        # Add all signal values (module-specific)
        for k, v in signals.items():
            if k not in template_vars:
                template_vars[k] = v

        # Fill template — use .format_map with defaultdict for missing keys
        from collections import defaultdict

        safe_vars = defaultdict(lambda: "N/A", template_vars)
        try:
            prompt = template.format_map(safe_vars)
        except Exception:
            logger.exception("Failed to format prompt for module '%s'", module_name)
            return None

        # Call LLM
        try:
            raw_response = self._client.generate(prompt, system=_SYSTEM_PROMPT)
        except Exception:
            logger.exception("LLM generation failed for module '%s'", module_name)
            return None

        # Parse JSON response
        result = self._parse_response(raw_response, signals)
        return result

    def _parse_response(self, raw: str, signals: dict) -> JudgmentResult:
        """Parse LLM JSON response into JudgmentResult."""
        parsed = None

        # Strategy 1: direct JSON parse
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract from markdown code fence
        if parsed is None:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # Strategy 3: extract first {...} block
        if parsed is None:
            match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        # Strategy 4: try to find nested JSON (handles nested arrays/objects)
        if parsed is None:
            # Find the first { and try progressively larger substrings
            start = raw.find("{")
            if start >= 0:
                for end in range(len(raw), start, -1):
                    if raw[end - 1] == "}":
                        try:
                            parsed = json.loads(raw[start:end])
                            break
                        except json.JSONDecodeError:
                            continue

        if parsed is None:
            logger.warning("Could not parse LLM response as JSON: %s", raw[:200])
            return JudgmentResult(
                reasoning="LLM response could not be parsed.",
                raw_llm_response=raw,
                module_signals=signals,
            )

        # Validate verdict
        verdict = parsed.get("verdict", "needs_verification")
        if verdict not in ("flagged", "pass", "needs_verification"):
            verdict = "needs_verification"

        # Validate severity
        severity = parsed.get("severity", "medium")
        if severity not in ("high", "medium", "low"):
            severity = "medium"

        # Validate confidence
        try:
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        return JudgmentResult(
            verdict=verdict,
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
            highlight_spans=parsed.get("highlight_spans", []),
            suggestion=parsed.get("suggestion", ""),
            evidence_used=parsed.get("evidence_used", []),
            severity=severity,
            raw_llm_response=raw,
            module_signals=signals,
        )
```

- [ ] **Step 2: Test LLMJudge instantiation and template formatting**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
from pipeline.llm.judge import LLMJudge, JudgmentResult, PROMPT_TEMPLATES
judge = LLMJudge()
print(f'LLMJudge created, available={judge.is_available()}')
print(f'Templates: {list(PROMPT_TEMPLATES.keys())}')
assert len(PROMPT_TEMPLATES) == 7, f'Expected 7 templates, got {len(PROMPT_TEMPLATES)}'

# Test JudgmentResult
jr = JudgmentResult(verdict='flagged', confidence=0.9, reasoning='Test')
d = jr.to_dict()
assert d['verdict'] == 'flagged'
print('JudgmentResult OK')

# Test JSON parsing
result = judge._parse_response('{\"verdict\": \"flagged\", \"confidence\": 0.85, \"reasoning\": \"test reason\", \"highlight_spans\": [], \"suggestion\": \"\", \"evidence_used\": [], \"severity\": \"high\"}', {})
assert result.verdict == 'flagged'
assert result.confidence == 0.85
print('JSON parsing OK')

# Test markdown fence parsing
result2 = judge._parse_response('Here is my analysis:\n\`\`\`json\n{\"verdict\": \"pass\", \"confidence\": 0.7, \"reasoning\": \"looks good\", \"highlight_spans\": [], \"suggestion\": \"\", \"evidence_used\": [], \"severity\": \"low\"}\n\`\`\`', {})
assert result2.verdict == 'pass'
print('Markdown fence parsing OK')

print('All LLMJudge tests passed')
"`

Expected: All tests passed

- [ ] **Step 3: Test LLMJudge with a real LLM call (if available)**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
from pipeline.llm.judge import LLMJudge
judge = LLMJudge()
if not judge.is_available():
    print('SKIP: No LLM available (Ollama not running, no GROQ_API_KEY)')
else:
    result = judge.judge_claim(
        module_name='vague_claims',
        claim_text='We are committed to significantly reducing our environmental footprint.',
        signals={
            'page': 5,
            'section_path': ['Environment', 'Climate'],
            'specificity_score': 0.2,
            'commitment_label': 'commitment',
            'zeroshot_result': 'vague aspirational claim: 0.8',
            'vague_terms': ['significantly', 'committed to'],
            'positive_signals': 'none',
            'key_phrases': ['environmental footprint', 'reducing'],
        },
    )
    if result:
        print(f'Verdict: {result.verdict}')
        print(f'Confidence: {result.confidence}')
        print(f'Reasoning: {result.reasoning}')
        print(f'Severity: {result.severity}')
        print(f'Suggestion: {result.suggestion}')
        print('LLM Judge real call OK')
    else:
        print('LLM returned None')
"`

Expected: A verdict with reasoning (or SKIP if no LLM available)

---

## Chunk 2: Upgrade Existing Modules 1-4 (Tasks 5-8)

### Task 5: Upgrade Module 1 — Vague Claims

**Files:**
- Modify: `backend/pipeline/modules/vague_claims.py`

Changes: Replace BART with DeBERTa zero-shot, add KeyBERT, integrate LLM Judge, remove Explainer.

- [ ] **Step 1: Read current vague_claims.py to understand structure**

Read: `backend/pipeline/modules/vague_claims.py` — note where BART model is loaded (look for `facebook/bart-large-mnli`), where Explainer is imported, and the analyze() method structure.

- [ ] **Step 2: Replace BART import with DeBERTa zero-shot config constant**

In `vague_claims.py`, replace the BART model loading with the new ZEROSHOT_MODEL from config. Find where `facebook/bart-large-mnli` is referenced and replace with:

```python
from config import ZEROSHOT_MODEL
```

Then where the model is loaded (in `_load_resources` or similar), change:
```python
self._zeroshot = hf_pipeline(
    "zero-shot-classification",
    model=ZEROSHOT_MODEL,
)
```

- [ ] **Step 3: Add KeyBERT loading**

In the `_load_resources` method, add:

```python
# KeyBERT for key phrase extraction
if self._keybert is None:
    try:
        from keybert import KeyBERT
        self._keybert = KeyBERT(model="all-MiniLM-L6-v2")
        logger.info("Loaded KeyBERT model")
    except Exception as e:
        logger.warning("KeyBERT not available: %s", e)
        self._keybert = None
```

Add `self._keybert = None` to `__init__`.

- [ ] **Step 4: Add a helper method to extract key phrases**

```python
def _extract_key_phrases(self, text: str) -> list[str]:
    """Extract top key phrases using KeyBERT."""
    if self._keybert is None:
        return []
    try:
        keywords = self._keybert.extract_keywords(
            text, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=5
        )
        return [kw[0] for kw in keywords]
    except Exception:
        return []
```

- [ ] **Step 5: Integrate LLM Judge into analyze()**

In the `analyze()` method, after gathering all signals (specificity score, commitment label, zero-shot result, vague terms, positive signals), add LLM Judge call:

```python
# --- LLM Judge ---
llm_judge = kwargs.get("llm_judge")
judgment_dict = None

if llm_judge is not None:
    key_phrases = self._extract_key_phrases(claim_text)
    signals = {
        "page": claim.page,
        "section_path": getattr(claim, "section_path", []),
        "specificity_score": round(specificity_score, 3),
        "commitment_label": commitment_label,
        "zeroshot_result": zeroshot_summary,
        "vague_terms": ", ".join(matched_vague) if matched_vague else "none",
        "positive_signals": ", ".join(positive_signal_details) if positive_signal_details else "none",
        "key_phrases": ", ".join(key_phrases) if key_phrases else "none",
    }

    evidence_for_judge = kwargs.get("evidence", [])
    kb_ctx = {"regulatory": kwargs.get("regulatory_evidence", [])}

    jr = llm_judge.judge_claim(
        module_name=self.name,
        claim_text=claim_text,
        signals=signals,
        evidence=evidence_for_judge,
        kb_context=kb_ctx,
    )
    if jr is not None:
        verdict_label = jr.verdict
        jr.module_signals = signals
        judgment_dict = jr.to_dict()
```

Then when creating the Verdict, pass `judgment=judgment_dict`.

- [ ] **Step 6: Remove Explainer import and usage**

Find and remove:
```python
from pipeline.llm.explainer import Explainer
explainer = Explainer()
```
And remove any `if explainer is not None:` blocks.

- [ ] **Step 7: Test Module 1 upgrade**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
from pipeline.modules.vague_claims import VagueClaimsModule
mod = VagueClaimsModule()
print(f'Module: {mod.name}, {mod.display_name}')
print('VagueClaimsModule loads OK')
"`

Expected: Module loads without errors

---

### Task 6: Upgrade Module 2 — No Proof

**Files:**
- Modify: `backend/pipeline/modules/no_proof.py`

Changes: Add DeBERTa NLI cross-encoder, add TCFD classifier, integrate LLM Judge, remove Explainer.

- [ ] **Step 1: Add DeBERTa NLI model loading**

In `_load_resources()`, add after the fact-check model loading:

```python
# DeBERTa NLI cross-encoder for stronger contradiction detection
if self._nli_model is None:
    try:
        from sentence_transformers import CrossEncoder
        from config import NLI_MODEL
        self._nli_model = CrossEncoder(NLI_MODEL)
        logger.info("Loaded NLI cross-encoder: %s", NLI_MODEL)
    except Exception as e:
        logger.warning("NLI cross-encoder not available: %s", e)
        self._nli_model = None
```

Add `self._nli_model = None` and `self._tcfd_model = None` to `__init__`.

- [ ] **Step 2: Add TCFD classifier loading**

```python
# TCFD category classifier
if self._tcfd_model is None:
    try:
        from transformers import pipeline as hf_pipeline
        from config import TCFD_MODEL
        self._tcfd_model = hf_pipeline(
            "text-classification", model=TCFD_MODEL, truncation=True
        )
        logger.info("Loaded TCFD model: %s", TCFD_MODEL)
    except Exception as e:
        logger.warning("TCFD model not available: %s", e)
        self._tcfd_model = None
```

- [ ] **Step 3: Add DeBERTa NLI evidence checking method**

```python
def _check_nli(self, claim_text: str, evidence_texts: list[str]) -> list[dict]:
    """Run DeBERTa NLI cross-encoder on (claim, evidence) pairs."""
    if self._nli_model is None or not evidence_texts:
        return []
    results = []
    try:
        pairs = [(claim_text, ev) for ev in evidence_texts]
        scores = self._nli_model.predict(pairs)
        labels = ["contradiction", "entailment", "neutral"]
        for score_set in scores:
            best_idx = score_set.argmax()
            results.append({
                "label": labels[best_idx],
                "scores": {l: float(s) for l, s in zip(labels, score_set)},
            })
    except Exception:
        logger.exception("DeBERTa NLI inference failed")
    return results
```

- [ ] **Step 4: Add TCFD classification method**

```python
def _classify_tcfd(self, text: str) -> str:
    """Classify text into TCFD category."""
    if self._tcfd_model is None:
        return "unknown"
    try:
        result = self._tcfd_model(text[:512])
        if isinstance(result, list):
            result = result[0]
        return result.get("label", "unknown")
    except Exception:
        return "unknown"
```

- [ ] **Step 5: Integrate LLM Judge into analyze()**

After gathering all signals (fact-check results, NLI results, checklist coverage, TCFD), add LLM Judge call following same pattern as Module 1 (Task 5 Step 5). Pass all signals including:
- `fact_check_results`: summary of ClimateBERT results
- `nli_results`: summary of DeBERTa NLI results
- `checklist_coverage`: percentage
- `missing_fields`: list
- `tcfd_category`: TCFD classification
- `evidence_count`: number of passages found

- [ ] **Step 6: Remove Explainer import and usage**

Same as Task 5 Step 6.

- [ ] **Step 7: Test Module 2 upgrade**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
from pipeline.modules.no_proof import NoProofModule
mod = NoProofModule()
print(f'Module: {mod.name}, {mod.display_name}')
print('NoProofModule loads OK')
"`

---

### Task 7: Upgrade Module 3 — Irrelevant Claims

**Files:**
- Modify: `backend/pipeline/modules/irrelevant_claims.py`

Changes: Add zero-shot classification, add Regulatory RAG, integrate LLM Judge. This is a MAJOR upgrade from pure regex.

- [ ] **Step 1: Read current irrelevant_claims.py**

Read the file to understand the current regex-only structure.

- [ ] **Step 2: Add zero-shot model loading**

Add to `_load_resources()`:

```python
if self._zeroshot is None:
    try:
        from transformers import pipeline as hf_pipeline
        from config import ZEROSHOT_MODEL
        self._zeroshot = hf_pipeline(
            "zero-shot-classification", model=ZEROSHOT_MODEL
        )
        logger.info("Loaded zero-shot model: %s", ZEROSHOT_MODEL)
    except Exception as e:
        logger.warning("Zero-shot model not available: %s", e)
        self._zeroshot = None
```

Add `self._zeroshot = None` to `__init__`.

- [ ] **Step 3: Add zero-shot irrelevance classification method**

```python
_IRRELEVANCE_LABELS = [
    "legally mandated requirement",
    "banned substance compliance",
    "industry standard practice",
    "voluntary environmental action",
]

def _classify_irrelevance(self, text: str) -> dict:
    """Zero-shot classify whether claim describes legally mandated action."""
    if self._zeroshot is None:
        return {}
    try:
        result = self._zeroshot(text, _IRRELEVANCE_LABELS, multi_label=False)
        return {
            label: round(score, 4)
            for label, score in zip(result["labels"], result["scores"])
        }
    except Exception:
        return {}
```

- [ ] **Step 4: Integrate into analyze() — keep regex as fast path, add zero-shot + LLM Judge**

After the regex check, for claims that DON'T match regex:
1. Run zero-shot classification
2. If "legally mandated" or "banned substance" score > 0.5, search Regulatory RAG
3. Package signals → LLM Judge

For claims that DO match regex: still run LLM Judge but include the regex match as a high-confidence signal.

- [ ] **Step 5: Test Module 3 upgrade**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
from pipeline.modules.irrelevant_claims import IrrelevantClaimsModule
mod = IrrelevantClaimsModule()
print(f'Module: {mod.name}, {mod.display_name}')
print('IrrelevantClaimsModule loads OK')
"`

---

### Task 8: Upgrade Module 4 — Lesser of Two Evils

**Files:**
- Modify: `backend/pipeline/modules/lesser_evil.py`

Changes: Add climate sentiment model, add climate stance model, integrate LLM Judge.

- [ ] **Step 1: Add sentiment and stance model loading**

In `_load_resources()`:

```python
# Climate sentiment
if self._sentiment_model is None:
    try:
        from transformers import pipeline as hf_pipeline
        from config import CLIMATE_SENTIMENT_MODEL
        self._sentiment_model = hf_pipeline(
            "text-classification", model=CLIMATE_SENTIMENT_MODEL, truncation=True
        )
        logger.info("Loaded sentiment model: %s", CLIMATE_SENTIMENT_MODEL)
    except Exception as e:
        logger.warning("Sentiment model not available: %s", e)
        self._sentiment_model = None

# Climate stance
if self._stance_model is None:
    try:
        from transformers import pipeline as hf_pipeline
        from config import CLIMATE_STANCE_MODEL
        self._stance_model = hf_pipeline(
            "text-classification", model=CLIMATE_STANCE_MODEL, truncation=True
        )
        logger.info("Loaded stance model: %s", CLIMATE_STANCE_MODEL)
    except Exception as e:
        logger.warning("Stance model not available: %s", e)
        self._stance_model = None
```

Add `self._sentiment_model = None` and `self._stance_model = None` to `__init__`.

- [ ] **Step 2: Add sentiment/stance analysis methods**

```python
def _analyze_sentiment(self, text: str) -> dict:
    if self._sentiment_model is None:
        return {"label": "unknown", "score": 0.0}
    try:
        out = self._sentiment_model(text[:512])
        if isinstance(out, list):
            out = out[0]
        return {"label": out["label"], "score": round(float(out["score"]), 4)}
    except Exception:
        return {"label": "unknown", "score": 0.0}

def _analyze_stance(self, text: str) -> dict:
    if self._stance_model is None:
        return {"label": "unknown", "score": 0.0}
    try:
        out = self._stance_model(text[:512])
        if isinstance(out, list):
            out = out[0]
        return {"label": out["label"], "score": round(float(out["score"]), 4)}
    except Exception:
        return {"label": "unknown", "score": 0.0}
```

- [ ] **Step 3: Integrate LLM Judge**

After gathering sector, risk level, buzzwords, sentiment, stance — package into signals dict and call LLM Judge.

- [ ] **Step 4: Test Module 4 upgrade**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
from pipeline.modules.lesser_evil import LesserEvilModule
mod = LesserEvilModule()
print(f'Module: {mod.name}, {mod.display_name}')
print('LesserEvilModule loads OK')
"`

---

## Chunk 3: Build New Modules 5-7 (Tasks 9-12)

### Task 9: Create `legitimate_labels.json` knowledge base

**Files:**
- Create: `backend/data/legitimate_labels.json`

- [ ] **Step 1: Create the knowledge base file**

Create `backend/data/legitimate_labels.json` with ~100+ legitimate eco-certifications. Structure:

```json
{
  "version": "1.0.0",
  "labels": [
    {
      "name": "Forest Stewardship Council (FSC)",
      "short_names": ["FSC", "FSC Certified", "FSC Mix", "FSC 100%"],
      "issuing_body": "Forest Stewardship Council",
      "category": "forestry",
      "what_it_certifies": "Responsible forest management and chain of custody"
    }
  ]
}
```

Include these categories:
- **Forestry/Wood:** FSC, PEFC, SFI
- **Energy:** Energy Star, Green-e, EPEAT
- **Building:** LEED, BREEAM, Green Globes, WELL
- **Agriculture/Food:** USDA Organic, Fair Trade, Rainforest Alliance, UTZ, GlobalG.A.P.
- **Textiles:** GOTS, OEKO-TEX, Bluesign, Better Cotton (BCI)
- **Marine:** MSC, ASC
- **General Environmental:** EU Ecolabel, Blue Angel, Nordic Swan, Green Seal, EcoLogo
- **Carbon/Climate:** Gold Standard, Verra (VCS), CDP, SBTi, Climate Neutral
- **ISO Standards:** ISO 14001, ISO 14064, ISO 50001, ISO 14020, ISO 14024, ISO 14025
- **Corporate/ESG:** B Corp, GRI, TCFD, SASB, UN Global Compact
- **Chemicals/Materials:** Cradle to Cradle, REACH compliance, RoHS
- **Jewelry/Mining:** Responsible Jewellery Council, Fairmined
- **Electronics:** TCO Certified, TUV Rheinland Green Product
- **Packaging:** How2Recycle, OK Compost, Seedling (TUV)
- **Water:** Alliance for Water Stewardship (AWS), WaterSense

- [ ] **Step 2: Verify JSON is valid**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
import json
with open('data/legitimate_labels.json') as f:
    data = json.load(f)
labels = data['labels']
print(f'Loaded {len(labels)} legitimate labels')
assert len(labels) >= 80, f'Expected 80+, got {len(labels)}'
# Check structure
for l in labels[:3]:
    assert 'name' in l
    assert 'short_names' in l
    assert 'issuing_body' in l
    print(f'  - {l[\"name\"]} ({len(l[\"short_names\"])} aliases)')
print('KB structure OK')
"`

---

### Task 10: Build Module 5 — Hidden Tradeoffs

**Files:**
- Create: `backend/pipeline/modules/hidden_tradeoffs.py`

- [ ] **Step 1: Create hidden_tradeoffs.py**

Create `backend/pipeline/modules/hidden_tradeoffs.py` implementing `HiddenTradeoffsModule(BaseModule)` with:

- `name = "hidden_tradeoffs"`, `display_name = "Hidden Tradeoffs"`
- `__init__`: init `_esg_model`, `_tcfd_model`, `_keybert`, all None
- `_load_resources()`: lazy-load ESG-BERT (reuse config.ESG_TOPIC_MODEL), TCFD model, KeyBERT
- `_detect_scope_narrowing(text)`: regex for "our offices", "one facility", "packaging only", "in [country/region]", "headquarters", "this product line"
- `_get_sector_topics(sector_id)`: lookup `industry_risk.json` for expected_material_topics
- `_run_topic_coverage(document_topics, expected_topics)`: compare BERTopic discovered topics against expected material topics using sentence-transformers similarity
- `analyze(claims, **kwargs)`:
  1. Get sector from kwargs (cached from M4) or run ESG-BERT
  2. Get expected material topics from industry_risk.json
  3. Get document_topics from kwargs (BERTopic result from app.py)
  4. For each claim: run TCFD, KeyBERT, scope narrowing regex
  5. LLM Judge with all signals
  6. Fallback: flag if >=50% expected topics missing AND claim focuses on non-material topic

- [ ] **Step 2: Test Module 5 loads**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
from pipeline.modules.hidden_tradeoffs import HiddenTradeoffsModule
mod = HiddenTradeoffsModule()
print(f'Module: {mod.name}, {mod.display_name}')
print('HiddenTradeoffsModule loads OK')
"`

---

### Task 11: Build Module 6 — Fake Labels

**Files:**
- Create: `backend/pipeline/modules/fake_labels.py`

- [ ] **Step 1: Create fake_labels.py**

Create `backend/pipeline/modules/fake_labels.py` implementing `FakeLabelsModule(BaseModule)` with:

- `name = "fake_labels"`, `display_name = "Fake Labels"`
- `__init__`: init `_labels_kb`, `_zeroshot`, `_st_model`, `_label_embeddings`, `_nlp`, all None
- `_load_resources()`: load legitimate_labels.json, zero-shot model, sentence-transformers, spaCy
- `_extract_certifications(text)`: regex for cert-like phrases — patterns:
  - `r"(?i)\b\w+[\s-]?(?:certified|certification|approved|accredited|verified|compliant|standard|label|seal|mark)\b"`
  - Trademark symbols: `r"[™®℠]"`
  - "meets X standard", "in accordance with X", "aligned with X"
- `_lookup_legitimate(cert_name)`: exact match against short_names in KB
- `_fuzzy_match(cert_name)`: cosine similarity against all label name embeddings
- `_classify_label_type(text)`: zero-shot with labels ["third-party verified certification", "self-awarded label", "industry standard", "marketing language"]
- `analyze(claims, **kwargs)`:
  1. Extract certification phrases
  2. For each: lookup legitimate → fuzzy match → zero-shot classify
  3. spaCy NER for ORG entities near cert language
  4. Regulatory RAG search
  5. LLM Judge
  6. Fallback: flag if not in KB and fuzzy < 0.8

- [ ] **Step 2: Test Module 6 loads**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
from pipeline.modules.fake_labels import FakeLabelsModule
mod = FakeLabelsModule()
print(f'Module: {mod.name}, {mod.display_name}')
print('FakeLabelsModule loads OK')
"`

---

### Task 12: Build Module 7 — Fibbing

**Files:**
- Create: `backend/pipeline/modules/fibbing.py`

- [ ] **Step 1: Create fibbing.py**

Create `backend/pipeline/modules/fibbing.py` implementing `FibbingModule(BaseModule)` with:

- `name = "fibbing"`, `display_name = "Fibbing"`
- `__init__`: init `_fact_check_model`, `_nli_model`, `_sentiment_model`, `_tapas_model`, `_tapas_tokenizer`, all None
- `_load_resources()`: lazy-load ClimateBERT fact-checker, DeBERTa NLI cross-encoder, climate sentiment, TAPAS
- `_detect_superlatives(text)`: regex for "first to", "only company", "100%", "zero emissions", "completely eliminated", "entirely", "never", "always", "all of our", "every single"
- `_check_contradictions(claim, evidence_texts)`: run DeBERTa NLI, return passages with contradiction score > 0.5
- `_check_facts(claim, evidence_texts)`: run ClimateBERT fact-checker (reuse from M2 pattern)
- `_check_sentiment_consistency(claim, evidence_texts)`: compare sentiment of claim vs evidence
- `_verify_tables(claim, tables)`: if TAPAS loaded and tables available, verify claim against tables
- `analyze(claims, **kwargs)`:
  1. Detect superlatives
  2. RAG retrieve passages about same topic
  3. ClimateBERT fact-check
  4. DeBERTa NLI contradiction detection
  5. Sentiment consistency
  6. TAPAS table verification (if table_data in kwargs)
  7. LLM Judge
  8. Fallback: flag if contradiction > 0.7 OR TAPAS refutes OR superlative with no support

- [ ] **Step 2: Test Module 7 loads**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
from pipeline.modules.fibbing import FibbingModule
mod = FibbingModule()
print(f'Module: {mod.name}, {mod.display_name}')
print('FibbingModule loads OK')
"`

---

## Chunk 4: Pipeline Integration + Testing (Tasks 13-15)

### Task 13: Wire everything together in app.py

**Files:**
- Modify: `backend/app.py`

- [ ] **Step 1: Add lazy-loaded getters for new components**

Add after existing getters:

```python
_hidden_tradeoffs_module = None
_fake_labels_module = None
_fibbing_module = None
_llm_judge = None

def get_hidden_tradeoffs_module():
    global _hidden_tradeoffs_module
    if _hidden_tradeoffs_module is None:
        from pipeline.modules.hidden_tradeoffs import HiddenTradeoffsModule
        _hidden_tradeoffs_module = HiddenTradeoffsModule()
    return _hidden_tradeoffs_module

def get_fake_labels_module():
    global _fake_labels_module
    if _fake_labels_module is None:
        from pipeline.modules.fake_labels import FakeLabelsModule
        _fake_labels_module = FakeLabelsModule()
    return _fake_labels_module

def get_fibbing_module():
    global _fibbing_module
    if _fibbing_module is None:
        from pipeline.modules.fibbing import FibbingModule
        _fibbing_module = FibbingModule()
    return _fibbing_module

def get_llm_judge():
    global _llm_judge
    if _llm_judge is None:
        from pipeline.llm.judge import LLMJudge
        _llm_judge = LLMJudge()
    return _llm_judge
```

- [ ] **Step 2: Update `/api/analyze` endpoint**

After extracting claims and before running modules, add BERTopic document-level analysis:

```python
# Document-level topic analysis for Module 5
document_topics = []
try:
    from bertopic import BERTopic
    all_texts = [e.text for e in elements if len(e.text) > 50]
    if len(all_texts) >= 5:  # BERTopic needs minimum documents
        topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", verbose=False)
        topics, _ = topic_model.fit_transform(all_texts)
        topic_info = topic_model.get_topic_info()
        document_topics = [
            topic_model.get_topic(t)
            for t in topic_info["Topic"].tolist()
            if t != -1
        ]
        print(f"[{report_id[:8]}] BERTopic found {len(document_topics)} topics")
except Exception as e:
    print(f"[{report_id[:8]}] BERTopic skipped: {e}")

# Collect table data for Module 7
table_data = [e for e in raw_elements if e.get("table_data")]
```

Get the shared LLM judge:

```python
judge = get_llm_judge()
```

Update all module calls to pass shared kwargs:

```python
# Module 1: Vague Claims
vague_verdicts = vague_mod.analyze(claims, llm_judge=judge)

# Module 2: No Proof
no_proof_verdicts = no_proof_mod.analyze(
    claims, retriever=retriever,
    regulatory_retriever=get_regulatory_retriever(),
    llm_judge=judge,
)

# Module 3: Irrelevant Claims
irr_verdicts = irr_mod.analyze(
    claims,
    regulatory_retriever=get_regulatory_retriever(),
    llm_judge=judge,
)

# Module 4: Lesser Evil
lesser_verdicts = lesser_mod.analyze(claims, llm_judge=judge)

# Module 5: Hidden Tradeoffs
hidden_mod = get_hidden_tradeoffs_module()
hidden_verdicts = hidden_mod.analyze(
    claims, retriever=retriever,
    document_topics=document_topics,
    llm_judge=judge,
)
all_verdicts.extend(hidden_verdicts)

# Module 6: Fake Labels
fake_mod = get_fake_labels_module()
fake_verdicts = fake_mod.analyze(
    claims,
    regulatory_retriever=get_regulatory_retriever(),
    llm_judge=judge,
)
all_verdicts.extend(fake_verdicts)

# Module 7: Fibbing
fib_mod = get_fibbing_module()
fib_verdicts = fib_mod.analyze(
    claims, retriever=retriever,
    table_data=table_data,
    llm_judge=judge,
)
all_verdicts.extend(fib_verdicts)
```

- [ ] **Step 3: Update `/api/analyze/text` endpoint with same changes**

Mirror the same module calls in the text endpoint.

- [ ] **Step 4: Test API starts without errors**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && timeout 15 python -c "
from app import app
print('FastAPI app created successfully')
print(f'Routes: {[r.path for r in app.routes]}')
"`

---

### Task 14: End-to-end test with NVIDIA ESG report

**Files:** None (testing only)

- [ ] **Step 1: Start the server**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && uvicorn app:app --host 0.0.0.0 --port 8000`

- [ ] **Step 2: Upload NVIDIA ESG report and analyze**

In a separate terminal:
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/ESG report/NVIDIA-Sustainability-Report-Fiscal-Year-2025.pdf" \
  > nvidia_results.json
```

- [ ] **Step 3: Review results quality**

Run: `cd "C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/backend" && python -c "
import json
with open('nvidia_results.json') as f:
    data = json.load(f)
print(f'Total claims: {data[\"total_claims\"]}')
print(f'Total flagged: {data[\"total_flagged\"]}')
print()
for cat in data.get('categories', []):
    print(f'--- {cat[\"display_name\"]} ---')
    print(f'  Total: {cat[\"total_items\"]}, Flagged: {cat[\"flagged_count\"]}, Needs verify: {cat[\"needs_verification_count\"]}, Pass: {cat[\"pass_count\"]}')
    for item in cat['items'][:2]:
        print(f'  [{item[\"verdict\"]}] {item[\"claim_text\"][:80]}...')
        if item.get('judgment'):
            j = item['judgment']
            print(f'    Reasoning: {j.get(\"reasoning\", \"N/A\")[:120]}')
            print(f'    Confidence: {j.get(\"confidence\", \"?\")}, Severity: {j.get(\"severity\", \"?\")}')
    print()
"`

Check:
- All 7 categories appear in output
- Verdicts make qualitative sense
- LLM judgments have reasoning, confidence, severity
- No crashes or empty categories

---

### Task 15: Test with additional ESG reports

**Files:** None (testing only)

- [ ] **Step 1: Test with a high-risk sector report (Chevron)**

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/ESG report/Chevron_2024_Sustainability_Highlights.pdf" \
  > chevron_results.json
```

Expect: Module 4 (Lesser Evil) should flag more claims since Chevron is fossil fuels (very high risk). Module 5 (Hidden Tradeoffs) should identify missing material topics.

- [ ] **Step 2: Test with a consumer goods report (Coca-Cola)**

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/ESG report/Coca_Cola_2024_Environmental_Update.pdf" \
  > cocacola_results.json
```

Expect: Module 6 (Fake Labels) may find certification references to verify. Module 1 (Vague) should catch unquantified claims.

- [ ] **Step 3: Test with a finance report (Wells Fargo)**

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@C:/Users/owen3/OneDrive/Desktop/Warwick CS/Spark the Globe/Green_Lens/ESG report/Wells_Fargo_2024_Climate_Report.pdf" \
  > wellsfargo_results.json
```

- [ ] **Step 4: Compare results across sectors**

Review the three JSON outputs side by side. Each sector should show different module activation patterns:
- Fossil fuels → high M4 (lesser evil) and M5 (hidden tradeoffs)
- Consumer goods → more M1 (vague) and M6 (fake labels)
- Finance → more M2 (no proof) for climate commitments

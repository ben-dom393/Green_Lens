# Sin Scoring System Design

**Date**: 2026-03-22
**Status**: Approved for implementation

## 1. Purpose

Add a quantitative scoring layer on top of the existing LLM Judge verdicts. Each claim gets scored against the 7 Sins of Greenwashing using explicit signal criteria (0/0.5/1 per signal), producing auditable numeric scores (0-100 per sin) and a combined claim risk score.

## 2. Architecture

```
Stage 1 (existing)              Stage 2 (new)                    Output
──────────────────              ─────────────                    ──────
Claim → 7 Modules               Per-sin Scorer LLM              SinScoreResult
  ↓                              (7 Groq calls per claim)         - 7 sin scores (0-100)
  7 JudgmentResults              Each call:                       - signal breakdowns
  (verdict, reasoning,            - receives claim text            - claim_risk (0-100)
   suggestion, highlight,         - receives Stage 1 judgment      - top_drivers
   evidence_used)                 - outputs 0/0.5/1 per signal
                                  Python computes weighted scores
```

## 3. Scoring Scale

Each signal within a sin is scored on a 3-point scale:

| Value | Meaning |
|-------|---------|
| 0     | No risk signal observed |
| 0.5   | Partial or ambiguous signal |
| 1     | Clear risk signal observed |

Sin Score formula: `Sin_Score_j = 100 × Σ(weight_jk × signal_value_jk)`

Each sin score ranges 0-100.

## 4. Signal Definitions and Weights

### 4.1 Hidden Tradeoff
| Signal | Weight | Description |
|--------|--------|-------------|
| selective_positive_focus | 0.35 | Claim presents benefit on favourable dimension only |
| omitted_negative_impact | 0.30 | Material tradeoff exists but not disclosed |
| partial_scope_as_whole | 0.20 | Claim covers subset but reads as company-wide |
| lifecycle_gap | 0.15 | Ignores relevant lifecycle stage |

### 4.2 No Proof
| Signal | Weight | Description |
|--------|--------|-------------|
| missing_metric | 0.30 | No quantified result, baseline, or KPI |
| missing_methodology | 0.25 | No calculation approach, boundary, scope, baseline year |
| missing_traceable_source | 0.20 | No footnote, appendix, data table, or evidence trail |
| missing_assurance | 0.15 | No independent verification or third-party certification |
| timing_mismatch | 0.10 | Evidence refers to inconsistent period or outdated data |

### 4.3 Vagueness
| Signal | Weight | Description |
|--------|--------|-------------|
| undefined_broad_terms | 0.35 | Uses "green", "sustainable", "eco-friendly" without criteria |
| absolute_without_qualifier | 0.30 | Uses "all", "zero", "neutral" without conditions/boundaries |
| scope_ambiguity | 0.20 | Cannot tell if company/product/site/campaign level |
| unsupported_positive_framing | 0.15 | Implies major benefit but support is narrow or procedural |

### 4.4 False Labels
| Signal | Weight | Description |
|--------|--------|-------------|
| unknown_issuer | 0.35 | Label/badge cannot be matched to credible certifier |
| no_verifiable_details | 0.25 | No certificate number, standard reference, validity info |
| misleading_iconography | 0.20 | Self-created icon resembles formal eco-label |
| scope_mismatch | 0.20 | Valid label but doesn't apply to claimed product/period/entity |

### 4.5 Irrelevance
| Signal | Weight | Description |
|--------|--------|-------------|
| legal_baseline | 0.40 | Feature is regulatory compliance, not additional achievement |
| market_norm | 0.25 | Feature is standard practice across the market |
| no_meaningful_differentiation | 0.20 | Even if true, wouldn't change investigation priority |
| immaterial_aspect | 0.15 | Concerns peripheral issue while larger impacts remain |

### 4.6 Lesser of Two Evils
| Signal | Weight | Description |
|--------|--------|-------------|
| relative_as_absolute | 0.30 | Implies "good" instead of "less harmful than X" |
| missing_comparator | 0.25 | No baseline stated for comparison |
| harmful_category_omitted | 0.25 | Ignores wider environmental profile of the category |
| exaggerated_significance | 0.20 | Minor change framed as major achievement |

### 4.7 Fibbing
| Signal | Weight | Description |
|--------|--------|-------------|
| internal_contradiction | 0.40 | Conflicts with figures/tables in same report |
| prior_contradiction | 0.25 | Conflicts with prior-period disclosures |
| third_party_contradiction | 0.20 | External source contradicts the claim |
| unsupported_factual | 0.15 | States factual outcome with no evidence |

## 5. Claim Risk Aggregation

```
Claim_Risk = min(100, Σ(α_j × Sin_Score_j))
```

Aggregation weights (from spec):

| Sin | Weight α |
|-----|----------|
| No Proof | 0.22 |
| Fibbing | 0.22 |
| Hidden Tradeoff | 0.16 |
| Vagueness | 0.14 |
| False Labels | 0.10 |
| Lesser of Two Evils | 0.10 |
| Irrelevance | 0.06 |

## 6. Implementation

### 6.1 New file: `pipeline/llm/scorer.py`

- `SinScoreResult` dataclass with all fields
- `LLMScorer` class with:
  - 7 prompt templates (one per sin)
  - `score_claim(claim_text, judgments: dict[str, JudgmentResult])` method
  - Calls Groq 7 times per claim (reuses existing `LLMClient` with throttling)
  - Parses signal values, computes sin scores via weighted formula
  - Computes claim_risk via aggregation weights
  - Identifies top_drivers (signals with value ≥ 0.5, sorted by weighted contribution)

### 6.2 Modified: `app.py`

- After Stage 1 (modules + judge), run Stage 2 (scorer) on each claim
- Attach `sin_scores` dict to each verdict item in the report output

### 6.3 Modified: `aggregator.py`

- Include `sin_scores`, `signal_breakdowns`, `claim_risk`, `top_drivers` in report output

## 7. Output Format (per claim in JSON response)

```json
{
  "claim_text": "We achieved 100% renewable electricity...",
  "verdict": "pass",
  "judgment": { ... Stage 1 output ... },
  "sin_scores": {
    "hidden_tradeoff": 35.0,
    "no_proof": 72.5,
    "vagueness": 47.5,
    "false_labels": 0.0,
    "irrelevance": 0.0,
    "lesser_of_two_evils": 15.0,
    "fibbing": 0.0
  },
  "signal_breakdowns": {
    "no_proof": {
      "missing_metric": 1,
      "missing_methodology": 0.5,
      "missing_traceable_source": 0.5,
      "missing_assurance": 1,
      "timing_mismatch": 0
    }
  },
  "claim_risk": 28.4,
  "top_drivers": ["no_proof: missing_metric", "no_proof: missing_assurance", "vagueness: undefined_broad_terms"]
}
```

## 8. Rate Limiting

- 7 scoring calls per claim × ~19 claims = ~133 calls
- Existing proactive throttle (28 calls/60s) handles this
- Expected scoring time: ~8-10 minutes for a 19-claim report
- Stage 1 + Stage 2 total: ~15-18 minutes per report

## 9. Constraints

- No materiality multiplier in this version
- No report-level priority formula — just claim-level sin scores and claim_risk
- Scoring is additive to existing pipeline — Stage 1 verdicts unchanged
- If scorer LLM fails for a claim, sin_scores = null (graceful degradation)

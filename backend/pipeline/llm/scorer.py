"""Stage 2 Sin Scorer: scores each claim against the 7 Sins using signal criteria."""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Weights from the ESG Disclosure Investigation Priority Score spec
# ------------------------------------------------------------------

SIN_SIGNAL_WEIGHTS: dict[str, dict[str, float]] = {
    "hidden_tradeoff": {
        "selective_positive_focus": 0.35,
        "omitted_negative_impact": 0.30,
        "partial_scope_as_whole": 0.20,
        "lifecycle_gap": 0.15,
    },
    "no_proof": {
        "missing_metric": 0.30,
        "missing_methodology": 0.25,
        "missing_traceable_source": 0.20,
        "missing_assurance": 0.15,
        "timing_mismatch": 0.10,
    },
    "vagueness": {
        "undefined_broad_terms": 0.35,
        "absolute_without_qualifier": 0.30,
        "scope_ambiguity": 0.20,
        "unsupported_positive_framing": 0.15,
    },
    "false_labels": {
        "unknown_issuer": 0.35,
        "no_verifiable_details": 0.25,
        "misleading_iconography": 0.20,
        "scope_mismatch": 0.20,
    },
    "irrelevance": {
        "legal_baseline": 0.40,
        "market_norm": 0.25,
        "no_meaningful_differentiation": 0.20,
        "immaterial_aspect": 0.15,
    },
    "lesser_of_two_evils": {
        "relative_as_absolute": 0.30,
        "missing_comparator": 0.25,
        "harmful_category_omitted": 0.25,
        "exaggerated_significance": 0.20,
    },
    "fibbing": {
        "internal_contradiction": 0.40,
        "prior_contradiction": 0.25,
        "third_party_contradiction": 0.20,
        "unsupported_factual": 0.15,
    },
}

# Claim risk aggregation weights (prioritises lack of evidence and fibbing)
CLAIM_RISK_WEIGHTS: dict[str, float] = {
    "no_proof": 0.22,
    "fibbing": 0.22,
    "hidden_tradeoff": 0.16,
    "vagueness": 0.14,
    "false_labels": 0.10,
    "lesser_of_two_evils": 0.10,
    "irrelevance": 0.06,
}

# Module name → sin name mapping (modules use different naming)
MODULE_TO_SIN: dict[str, str] = {
    "vague_claims": "vagueness",
    "no_proof": "no_proof",
    "irrelevant_claims": "irrelevance",
    "lesser_of_two_evils": "lesser_of_two_evils",
    "hidden_tradeoffs": "hidden_tradeoff",
    "fake_labels": "false_labels",
    "fibbing": "fibbing",
}

SIN_TO_MODULE: dict[str, str] = {v: k for k, v in MODULE_TO_SIN.items()}


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class SinScoreResult:
    """Quantitative scoring output for a single claim."""

    sin_scores: dict[str, float] = field(default_factory=dict)
    signal_breakdowns: dict[str, dict[str, float]] = field(default_factory=dict)
    claim_risk: float = 0.0
    top_drivers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# System prompt for scorer
# ------------------------------------------------------------------

_SCORER_SYSTEM = (
    "You are an ESG disclosure scoring analyst. You evaluate environmental "
    "claims against specific risk signal criteria. For each signal, assign "
    "exactly 0, 0.5, or 1. Respond with valid JSON only — no markdown, "
    "no extra text."
)

# ------------------------------------------------------------------
# Per-sin scoring prompt templates
# ------------------------------------------------------------------

_SIN_PROMPTS: dict[str, str] = {
    "hidden_tradeoff": """Score this claim's "Hidden Tradeoff" risk signals.

CLAIM: "{claim_text}"

ANALYSIS (from greenwashing detection module):
- Verdict: {verdict}
- Reasoning: {reasoning}
- Suggestion: {suggestion}

Score each signal as 0 (no risk), 0.5 (partial/ambiguous), or 1 (clear risk):

1. selective_positive_focus: The claim presents a benefit on only a favourable dimension (e.g., packaging recyclability without discussing upstream emissions).
2. omitted_negative_impact: A material environmental tradeoff exists in the same product/process but is not disclosed near the claim.
3. partial_scope_as_whole: The claim covers only a product line, facility, region, or pilot but reads as company-wide.
4. lifecycle_gap: The claim ignores a relevant lifecycle stage (extraction, processing, transport, use, or disposal).

Respond JSON only:
{{"selective_positive_focus": 0, "omitted_negative_impact": 0, "partial_scope_as_whole": 0, "lifecycle_gap": 0}}""",

    "no_proof": """Score this claim's "No Proof" risk signals.

CLAIM: "{claim_text}"

ANALYSIS (from greenwashing detection module):
- Verdict: {verdict}
- Reasoning: {reasoning}
- Suggestion: {suggestion}

Score each signal as 0 (no risk), 0.5 (partial/ambiguous), or 1 (clear risk):

1. missing_metric: No quantified result, baseline value, or supporting KPI is linked to the claim.
2. missing_methodology: No explanation of calculation approach, organisational boundary, scope, or baseline year.
3. missing_traceable_source: No footnote, appendix, data table, or other evidence trail can be found.
4. missing_assurance: No independent verification, third-party certification, or meaningful internal review is disclosed.
5. timing_mismatch: The evidence exists, but refers to an inconsistent reporting period or out-of-date measurement window.

Respond JSON only:
{{"missing_metric": 0, "missing_methodology": 0, "missing_traceable_source": 0, "missing_assurance": 0, "timing_mismatch": 0}}""",

    "vagueness": """Score this claim's "Vagueness" risk signals.

CLAIM: "{claim_text}"

ANALYSIS (from greenwashing detection module):
- Verdict: {verdict}
- Reasoning: {reasoning}
- Suggestion: {suggestion}

Score each signal as 0 (no risk), 0.5 (partial/ambiguous), or 1 (clear risk):

1. undefined_broad_terms: Uses words like "green", "sustainable", "eco-friendly", or "responsible" without clear criteria or definitions.
2. absolute_without_qualifier: Uses "all", "zero", "neutral", "net zero", "climate positive" without explaining conditions and boundaries.
3. scope_ambiguity: Readers cannot tell whether the claim is company-level, product-level, site-level, or campaign-level.
4. unsupported_positive_framing: The tone implies major environmental benefit while the actual support is narrow, partial, or procedural.

Respond JSON only:
{{"undefined_broad_terms": 0, "absolute_without_qualifier": 0, "scope_ambiguity": 0, "unsupported_positive_framing": 0}}""",

    "false_labels": """Score this claim's "False Labels" risk signals.

CLAIM: "{claim_text}"

ANALYSIS (from greenwashing detection module):
- Verdict: {verdict}
- Reasoning: {reasoning}
- Suggestion: {suggestion}

Score each signal as 0 (no risk), 0.5 (partial/ambiguous), or 1 (clear risk):

1. unknown_issuer: The label or badge cannot be matched to a credible standard setter, certifier, or registry.
2. no_verifiable_details: No certificate number, standard reference, issuer name, or validity information is provided.
3. misleading_iconography: A self-created icon or design element resembles a formal third-party eco-label.
4. scope_mismatch: A valid label is mentioned, but it doesn't apply to the product, period, entity, or location being claimed.

Respond JSON only:
{{"unknown_issuer": 0, "no_verifiable_details": 0, "misleading_iconography": 0, "scope_mismatch": 0}}""",

    "irrelevance": """Score this claim's "Irrelevance" risk signals.

CLAIM: "{claim_text}"

ANALYSIS (from greenwashing detection module):
- Verdict: {verdict}
- Reasoning: {reasoning}
- Suggestion: {suggestion}

Score each signal as 0 (no risk), 0.5 (partial/ambiguous), or 1 (clear risk):

1. legal_baseline: The feature highlighted is simply regulatory compliance rather than an additional environmental achievement.
2. market_norm: The claim emphasises a feature that is already standard practice across the market.
3. no_meaningful_differentiation: Even if true, the claim would not materially change the assessment of the report.
4. immaterial_aspect: The claim concerns a peripheral issue while larger unaddressed impacts remain more important.

Respond JSON only:
{{"legal_baseline": 0, "market_norm": 0, "no_meaningful_differentiation": 0, "immaterial_aspect": 0}}""",

    "lesser_of_two_evils": """Score this claim's "Lesser of Two Evils" risk signals.

CLAIM: "{claim_text}"

ANALYSIS (from greenwashing detection module):
- Verdict: {verdict}
- Reasoning: {reasoning}
- Suggestion: {suggestion}

Score each signal as 0 (no risk), 0.5 (partial/ambiguous), or 1 (clear risk):

1. relative_as_absolute: The claim implies "good for the environment" instead of "less harmful than baseline X".
2. missing_comparator: The report does not state compared with what, when, or by how much the product is supposedly greener.
3. harmful_category_omitted: The claim ignores the wider environmental profile of the category (e.g., fossil-fuel-intensive sector).
4. exaggerated_significance: A minor process change is framed as a major sustainability achievement.

Respond JSON only:
{{"relative_as_absolute": 0, "missing_comparator": 0, "harmful_category_omitted": 0, "exaggerated_significance": 0}}""",

    "fibbing": """Score this claim's "Fibbing" risk signals.

CLAIM: "{claim_text}"

ANALYSIS (from greenwashing detection module):
- Verdict: {verdict}
- Reasoning: {reasoning}
- Suggestion: {suggestion}

Score each signal as 0 (no risk), 0.5 (partial/ambiguous), or 1 (clear risk):

1. internal_contradiction: The claim conflicts with figures, tables, notes, or other sections in the same report.
2. prior_contradiction: The claim conflicts materially with prior-period disclosures without reconciliation.
3. third_party_contradiction: An external referenced source or official record contradicts the claim.
4. unsupported_factual: The text states a factual outcome as if proven, but no evidence exists and available data suggests otherwise.

Respond JSON only:
{{"internal_contradiction": 0, "prior_contradiction": 0, "third_party_contradiction": 0, "unsupported_factual": 0}}""",
}


# ------------------------------------------------------------------
# Scorer class
# ------------------------------------------------------------------

class LLMScorer:
    """Stage 2 scorer: converts LLM Judge verdicts into quantitative sin scores."""

    def __init__(self) -> None:
        self._client = LLMClient()

    def score_claim(
        self,
        claim_text: str,
        judgments: dict[str, dict],
    ) -> SinScoreResult | None:
        """Score a single claim across all 7 sins.

        Parameters
        ----------
        claim_text:
            The claim text to score.
        judgments:
            Dict mapping module_name → judgment dict (from Stage 1).
            Keys are module names like "vague_claims", "no_proof", etc.

        Returns
        -------
        SinScoreResult or None if scoring fails entirely.
        """
        if not self._client._get_api_key():
            logger.info("Scorer unavailable: no GROQ_API_KEY")
            return None

        all_signal_breakdowns: dict[str, dict[str, float]] = {}
        all_sin_scores: dict[str, float] = {}

        for sin_name, weights in SIN_SIGNAL_WEIGHTS.items():
            module_name = SIN_TO_MODULE.get(sin_name, sin_name)
            judgment = judgments.get(module_name, {})

            # Extract Stage 1 fields for the prompt
            verdict = judgment.get("verdict", "N/A")
            reasoning = judgment.get("reasoning", "No analysis available.")
            suggestion = judgment.get("suggestion", "N/A")

            # Get the prompt template
            template = _SIN_PROMPTS.get(sin_name)
            if not template:
                logger.warning("No scoring template for sin '%s'", sin_name)
                continue

            # Format prompt
            prompt = template.format(
                claim_text=claim_text[:500],
                verdict=verdict,
                reasoning=reasoning[:400],
                suggestion=suggestion[:300],
            )

            # Call LLM
            try:
                raw = self._client.generate(prompt, system=_SCORER_SYSTEM)
            except Exception:
                logger.exception("Scorer LLM call failed for sin '%s'", sin_name)
                continue

            if raw.startswith("[LLM unavailable"):
                logger.info("Scorer LLM unavailable for sin '%s'", sin_name)
                continue

            # Parse signal values
            signals = self._parse_signals(raw, weights)
            if signals is None:
                logger.warning(
                    "Could not parse scorer response for sin '%s': %s",
                    sin_name, raw[:200],
                )
                continue

            # Compute sin score: 100 × Σ(weight × signal_value)
            sin_score = 100.0 * sum(
                weights[sig] * signals.get(sig, 0)
                for sig in weights
            )
            sin_score = round(sin_score, 1)

            all_signal_breakdowns[sin_name] = signals
            all_sin_scores[sin_name] = sin_score

        if not all_sin_scores:
            return None

        # Compute claim risk: Σ(α_j × sin_score_j), capped at 100
        claim_risk_raw = sum(
            CLAIM_RISK_WEIGHTS.get(sin, 0) * score
            for sin, score in all_sin_scores.items()
        )
        claim_risk = round(min(100.0, claim_risk_raw), 1)

        # Identify top drivers: signals with value >= 0.5, sorted by contribution
        drivers: list[tuple[float, str]] = []
        for sin_name, signals in all_signal_breakdowns.items():
            weights = SIN_SIGNAL_WEIGHTS[sin_name]
            for sig_name, sig_value in signals.items():
                if sig_value >= 0.5:
                    contribution = weights.get(sig_name, 0) * sig_value * 100
                    drivers.append((contribution, f"{sin_name}: {sig_name}"))
        drivers.sort(reverse=True)
        top_drivers = [label for _, label in drivers[:5]]

        return SinScoreResult(
            sin_scores=all_sin_scores,
            signal_breakdowns=all_signal_breakdowns,
            claim_risk=claim_risk,
            top_drivers=top_drivers,
        )

    def _parse_signals(
        self, raw: str, expected_keys: dict[str, float]
    ) -> dict[str, float] | None:
        """Parse LLM response into signal values dict.

        Validates that all values are 0, 0.5, or 1.
        """
        parsed = None

        # Strategy 1: direct JSON
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract from code fence
        if parsed is None:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # Strategy 3: find first {...}
        if parsed is None:
            match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        if parsed is None or not isinstance(parsed, dict):
            return None

        # Validate and clamp signal values to {0, 0.5, 1}
        valid_values = {0, 0.0, 0.5, 1, 1.0}
        result: dict[str, float] = {}
        for key in expected_keys:
            val = parsed.get(key, 0)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0
            # Clamp to nearest valid value
            if val <= 0.25:
                val = 0
            elif val <= 0.75:
                val = 0.5
            else:
                val = 1
            result[key] = val

        return result

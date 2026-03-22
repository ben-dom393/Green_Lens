"""LLM Judge: makes final greenwashing verdicts using structured evidence from modules."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
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
# System prompt
# ------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a greenwashing analyst. You evaluate environmental claims "
    "from corporate ESG reports for signs of greenwashing. "
    "Give concise, factual assessments based on the evidence provided. "
    "Always respond with valid JSON only — no markdown, no extra text."
)

# ------------------------------------------------------------------
# Per-module prompt templates
# ------------------------------------------------------------------

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

{artifact_note}Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",

    "no_proof": """Evaluate whether this environmental claim has adequate supporting evidence in the document.

IMPORTANT GUIDELINES:
- "No Proof" means the company makes a green claim but provides NO evidence, data, or third-party verification to back it up.
- If evidence passages are found in the document that relate to the claim, the claim likely HAS some proof — even if the proof checklist coverage is low.
- Proof checklist coverage of 0% often means the automated system couldn't match the claim to a known claim type — it does NOT mean there is zero evidence. READ the actual evidence passages below to judge.
- Claims that reference specific standards (SBTi, ISO, LEED), provide numbers/percentages, or cite third-party verification generally have SOME proof and should PASS or get needs_verification.
- Only flag "no proof" if the claim makes a substantive assertion with genuinely NO supporting data, metrics, methodology, or verification anywhere in the document.
- A claim saying "we reduced emissions by 50%" with supporting evidence in the document = PASS, even if checklist coverage is 0%.

CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
- ClimateBERT fact-check: {fact_check_results}
- DeBERTa NLI: {nli_results}
- Proof checklist coverage: {checklist_coverage} (NOTE: 0% may mean claim type not matched, not absence of evidence)
- Missing evidence fields: {missing_fields}
- TCFD category: {tcfd_category}
- Evidence passages found: {evidence_count}

EVIDENCE FROM DOCUMENT (read these carefully):
{evidence}

REGULATORY CONTEXT:
{regulatory}

{artifact_note}Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",

    "irrelevant_claims": """Evaluate whether this environmental claim is greenwashing because it highlights something that is ALREADY LEGALLY REQUIRED or involves a SUBSTANCE ALREADY BANNED BY LAW.

IMPORTANT DISTINCTION:
- "Irrelevant" in the Seven Sins of Greenwashing means claiming credit for something the company MUST do by law anyway (e.g., "we don't use CFCs" when CFCs are banned, or "we comply with EU emissions standards" as if it's voluntary).
- Voluntary industry standards like SBTi, ISO certifications, or best practices are NOT irrelevant — companies choose to adopt them. These should generally PASS.
- Only flag claims where the company is taking credit for legal compliance as if it were a voluntary environmental achievement.

CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
- Regex KB match: {regex_match}
- Zero-shot classification: {zeroshot_result}

REGULATORY CONTEXT:
{regulatory}

{artifact_note}Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",

    "lesser_of_two_evils": """Evaluate whether this green claim is a "lesser of two evils" greenwashing tactic — promoting a relatively minor environmental benefit to DISTRACT from the company's much larger environmental harms.

IMPORTANT GUIDELINES:
- This is NOT about whether a claim is vague or unproven (other modules check that).
- A claim is "lesser of two evils" ONLY if it deliberately highlights a trivial green action while the company's core business causes significant, unaddressed environmental damage.
- Companies in ANY sector can legitimately report on their sustainability efforts. Reporting on renewable energy, waste reduction, or emissions targets is NORMAL corporate sustainability reporting, not distraction.
- Only flag if the claim is clearly disproportionate — e.g., an oil company emphasizing biodegradable coffee cups while ignoring fossil fuel emissions.
- Claims with specific metrics, third-party verification, or addressing material topics for their sector should generally PASS.
- Non-environmental claims (diversity, cybersecurity) appearing in an environmental section may be irrelevant but are NOT "lesser of two evils."

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

{artifact_note}Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",

    "hidden_tradeoffs": """Evaluate whether this claim hides a significant environmental tradeoff.

IMPORTANT GUIDELINES:
- "Hidden Tradeoff" means the claim highlights ONE environmental benefit while concealing a RELATED environmental cost. For example: "our product uses recycled packaging" while the manufacturing process produces toxic waste.
- This is about TRADEOFFS WITHIN THE CLAIM'S OWN SCOPE, not about whether the overall report covers all topics. A single claim does NOT need to address every material topic — that's the report's job, not each claim's.
- Only flag if the claim ACTIVELY CONCEALS a tradeoff directly related to what it discusses.
- Scope-narrowing language (e.g., "at our headquarters", "in select regions", "pilot program") may indicate hidden tradeoffs IF it obscures a larger unaddressed impact.
- A claim about renewable energy is NOT hiding tradeoffs just because the company also has e-waste issues — those are separate topics.
- When in doubt, PASS. Most legitimate sustainability claims are not hiding tradeoffs.

CLAIM: "{claim_text}"
PAGE: {page} | SECTION: {section_path}

MODEL SIGNALS:
- Sector: {sector}
- Document-level topic coverage: {found_topics} out of {expected_topics} material topics addressed in this report
- Missing material topics (document-level, NOT this claim's responsibility): {missing_topics}
- TCFD coverage: {tcfd_coverage}
- Claim focuses on: {claim_focus}
- Scope narrowing detected: {scope_narrowing}

EVIDENCE FROM DOCUMENT:
{evidence}

{artifact_note}Respond ONLY with valid JSON:
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

{artifact_note}Respond ONLY with valid JSON:
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

{artifact_note}Respond ONLY with valid JSON:
{{"verdict": "flagged" or "pass" or "needs_verification", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation", "highlight_spans": ["exact text spans to highlight"], "suggestion": "what would make this claim acceptable", "evidence_used": ["which evidence influenced decision"], "severity": "high" or "medium" or "low"}}""",
}


class LLMJudge:
    """Central LLM judge for all greenwashing detection modules."""

    def __init__(self) -> None:
        self._client = LLMClient()
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if Groq API key is set."""
        if self._available is None:
            self._available = bool(os.environ.get("GROQ_API_KEY"))
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
                text = ev.get("text", str(ev))[:1000]
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
                txt = r.get("text", "")[:500]
                reg_parts.append(f"[{doc}, p.{pg}] {txt}")
            regulatory_text = "\n".join(reg_parts)

        # Build artifact advisory note (if any artifact signals present)
        artifact_signals = signals.get("artifact_signals", [])
        if artifact_signals:
            artifact_note = (
                "NOTE: The claim text may contain PDF parsing artifacts "
                f"({', '.join(artifact_signals)}). These are noise from "
                "PDF extraction — ignore them when evaluating the claim's "
                "substance, but note if they make the claim hard to assess.\n\n"
            )
        else:
            artifact_note = ""

        # Merge all template variables
        template_vars = {
            "claim_text": claim_text,
            "page": signals.get("page", "?"),
            "section_path": " > ".join(signals.get("section_path", [])) or "N/A",
            "evidence": evidence_text,
            "regulatory": regulatory_text,
            "artifact_note": artifact_note,
        }
        # Add all signal values (module-specific)
        for k, v in signals.items():
            if k not in template_vars:
                template_vars[k] = v

        # Fill template with safe defaults for missing keys
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

        # Check for LLM unavailable placeholder
        if raw_response.startswith("[LLM unavailable"):
            logger.info("LLM unavailable for module '%s'", module_name)
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

        # Strategy 3: extract first simple {...} block
        if parsed is None:
            match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        # Strategy 4: find nested JSON (handles arrays/objects inside)
        if parsed is None:
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

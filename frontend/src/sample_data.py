from src.config import METRIC_NAMES

MOCK_COMPANY_NAME = "Verdant Axis plc"
MOCK_REPORT_TITLE = "2025 Sustainability Progress Update"
MOCK_PAGE_COUNT = 48


METRIC_EXPLANATIONS = {
    "Hidden Trade-Off": "The report highlights a narrow environmental win while downplaying adjacent impacts elsewhere in the value chain.",
    "No Proof": "Several claims gesture at progress, but supporting data, methodology, or external assurance are not clearly provided.",
    "Vagueness": "Terms such as sustainable, cleaner, and responsible appear often without enough operational specificity.",
    "False Labels": "References to standards and badges are present, but the basis and legitimacy of those signals are not always obvious.",
    "Irrelevance": "Some highlighted environmental attributes are technically true but not materially useful for judging overall sustainability.",
    "Lesser of Two Evils": "The narrative frames improvements within a fundamentally high-impact activity as if they resolve the broader issue.",
    "Fibbing": "A few statements read as overly absolute and would need careful verification before being treated as fact.",
}

EVIDENCE_SNIPPETS = [
    {
        "page": 3,
        "metric": "Hidden Trade-Off",
        "claim_text": "Our recyclable packaging program has reduced waste by 15% across our product lines.",
        "explanation": "The report celebrates recyclable packaging upgrades while offering little detail on the emissions increase from expedited logistics.",
        "verdict": "flagged",
    },
    {
        "page": 7,
        "metric": "No Proof",
        "claim_text": "We source 80% of our materials from low-carbon suppliers.",
        "explanation": "A statement on low-carbon sourcing is presented prominently, but no supporting dataset or assurance note is shown nearby.",
        "verdict": "flagged",
    },
    {
        "page": 11,
        "metric": "Vagueness",
        "claim_text": "Our operations are eco-conscious and aligned with sustainable development principles.",
        "explanation": "The company repeatedly describes its operations as eco-conscious without explaining what criteria qualify that label.",
        "verdict": "flagged",
    },
    {
        "page": 18,
        "metric": "False Labels",
        "claim_text": "This product line carries our Green Certified seal of approval.",
        "explanation": "A green-style seal appears beside a product line, but the certifying body and audit standard are not clearly identified.",
        "verdict": "flagged",
    },
    {
        "page": 24,
        "metric": "Irrelevance",
        "claim_text": "All our aerosol products are 100% CFC-free.",
        "explanation": "The report emphasizes that one product is CFC-free, even though that feature is already expected and not decision-useful.",
        "verdict": "needs_verification",
    },
    {
        "page": 29,
        "metric": "Lesser of Two Evils",
        "claim_text": "Our new low-emission fuel blend reduces carbon output by 10% compared to standard options.",
        "explanation": "A reduced-emission fossil fuel offering is marketed as a climate-positive choice despite the category remaining materially harmful.",
        "verdict": "flagged",
    },
    {
        "page": 36,
        "metric": "Fibbing",
        "claim_text": "We have achieved near-zero environmental impact across our global operations.",
        "explanation": "One headline claim implies near-zero impact, but the surrounding disclosures suggest the statement is too absolute to trust as written.",
        "verdict": "flagged",
    },
]

DEFAULT_METRIC_SCORES = dict(
    zip(
        METRIC_NAMES,
        [37, 31, 42, 27, 29, 38, 41],
    )
)

MOCK_RAW_TEXT_PREVIEW = """
Verdant Axis plc states that sustainability is fully embedded across operations and supplier relationships.
The report highlights emissions intensity improvements, circular packaging trials, and renewable power procurement.

Several environmental claims are framed in strong aspirational language, while some quantitative baselines remain lightly specified.
Independent assurance is referenced for selected indicators, but broader claim verification appears uneven across sections.

The narrative is intentionally placeholder text for UI design and should later be replaced with extracted report passages and model outputs.
""".strip()

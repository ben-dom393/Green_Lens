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
        "snippet": "The report celebrates recyclable packaging upgrades while offering little detail on the emissions increase from expedited logistics.",
    },
    {
        "page": 7,
        "metric": "No Proof",
        "snippet": "A statement on low-carbon sourcing is presented prominently, but no supporting dataset or assurance note is shown nearby.",
    },
    {
        "page": 11,
        "metric": "Vagueness",
        "snippet": "The company repeatedly describes its operations as eco-conscious without explaining what criteria qualify that label.",
    },
    {
        "page": 18,
        "metric": "False Labels",
        "snippet": "A green-style seal appears beside a product line, but the certifying body and audit standard are not clearly identified.",
    },
    {
        "page": 24,
        "metric": "Irrelevance",
        "snippet": "The report emphasizes that one product is CFC-free, even though that feature is already expected and not decision-useful.",
    },
    {
        "page": 29,
        "metric": "Lesser of Two Evils",
        "snippet": "A reduced-emission fossil fuel offering is marketed as a climate-positive choice despite the category remaining materially harmful.",
    },
    {
        "page": 36,
        "metric": "Fibbing",
        "snippet": "One headline claim implies near-zero impact, but the surrounding disclosures suggest the statement is too absolute to trust as written.",
    },
]

DEFAULT_METRIC_SCORES = dict(
    zip(
        METRIC_NAMES,
        [3.7, 3.1, 4.2, 2.7, 2.9, 3.8, 4.1],
    )
)

MOCK_RAW_TEXT_PREVIEW = """
Verdant Axis plc states that sustainability is fully embedded across operations and supplier relationships.
The report highlights emissions intensity improvements, circular packaging trials, and renewable power procurement.

Several environmental claims are framed in strong aspirational language, while some quantitative baselines remain lightly specified.
Independent assurance is referenced for selected indicators, but broader claim verification appears uneven across sections.

The narrative is intentionally placeholder text for UI design and should later be replaced with extracted report passages and model outputs.
""".strip()

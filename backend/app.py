"""Green Lens Backend — Greenwashing Detection API."""

import sys
import json
import uuid
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import DATA_DIR

app = FastAPI(title="Green Lens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for reports (MVP — no database)
reports_store: dict[str, dict] = {}


# ── Pipeline components (lazy-loaded) ──────────────────────────────────

_claim_extractor = None
_vague_module = None
_no_proof_module = None
_irrelevant_module = None
_lesser_evil_module = None
_hidden_tradeoffs_module = None
_fake_labels_module = None
_fibbing_module = None
_aggregator = None
_indexer = None
_retriever = None
_regulatory_indexer = None
_regulatory_retriever = None
_llm_judge = None


def get_claim_extractor():
    global _claim_extractor
    if _claim_extractor is None:
        from pipeline.claim_extractor import ClaimExtractor
        _claim_extractor = ClaimExtractor()
    return _claim_extractor


def get_vague_module():
    global _vague_module
    if _vague_module is None:
        from pipeline.modules.vague_claims import VagueClaimsModule
        _vague_module = VagueClaimsModule()
    return _vague_module


def get_no_proof_module():
    global _no_proof_module
    if _no_proof_module is None:
        from pipeline.modules.no_proof import NoProofModule
        _no_proof_module = NoProofModule()
    return _no_proof_module


def get_irrelevant_module():
    global _irrelevant_module
    if _irrelevant_module is None:
        from pipeline.modules.irrelevant_claims import IrrelevantClaimsModule
        _irrelevant_module = IrrelevantClaimsModule()
    return _irrelevant_module


def get_lesser_evil_module():
    global _lesser_evil_module
    if _lesser_evil_module is None:
        from pipeline.modules.lesser_evil import LesserEvilModule
        _lesser_evil_module = LesserEvilModule()
    return _lesser_evil_module


def get_aggregator():
    global _aggregator
    if _aggregator is None:
        from pipeline.modules.aggregator import Aggregator
        _aggregator = Aggregator()
    return _aggregator


def get_indexer():
    global _indexer
    if _indexer is None:
        from pipeline.rag.indexer import DocumentIndexer
        _indexer = DocumentIndexer()
    return _indexer


def get_retriever():
    global _retriever
    if _retriever is None:
        from pipeline.rag.retriever import HybridRetriever
        _retriever = HybridRetriever(get_indexer())
    return _retriever


def get_regulatory_indexer():
    global _regulatory_indexer
    if _regulatory_indexer is None:
        from pipeline.rag.regulatory_indexer import RegulatoryIndexer
        _regulatory_indexer = RegulatoryIndexer()
    return _regulatory_indexer


def get_regulatory_retriever():
    global _regulatory_retriever
    if _regulatory_retriever is None:
        from pipeline.rag.regulatory_retriever import RegulatoryRetriever
        _regulatory_retriever = RegulatoryRetriever(get_regulatory_indexer())
    return _regulatory_retriever


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


# ── Helper: parse PDF text ──────────────────────────────────────────────

def simple_text_extract(pdf_path: str) -> list[dict]:
    """Extract text and tables from a PDF using the general-purpose parser.

    Returns list of dicts matching DocumentElement structure with keys:
    element_id, text, page, element_type, section_path, and table_data.
    """
    try:
        from pipeline.pdf_parser import PDFParser
        parser = PDFParser()
        return parser.parse(pdf_path)
    except Exception as exc:
        # Fallback: return a notice so callers know what happened.
        return [
            {
                "element_id": "notice_0",
                "text": f"PDF parsing failed: {exc}",
                "page": 0,
                "element_type": "NarrativeText",
                "section_path": [],
                "table_data": None,
            }
        ]


# ── API Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/analyze")
async def analyze_report(file: UploadFile = File(...)):
    """Upload an ESG report PDF and run greenwashing detection."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        report_id = str(uuid.uuid4())

        # Step 1: Parse PDF → document elements
        print(f"[{report_id[:8]}] Parsing PDF: {file.filename}")
        raw_elements = simple_text_extract(tmp_path)
        if not raw_elements or raw_elements[0].get("element_id") == "notice_0":
            raise HTTPException(
                status_code=500,
                detail="PDF parsing failed. Install pymupdf: pip install pymupdf",
            )
        print(f"[{report_id[:8]}] Extracted {len(raw_elements)} text elements")

        # Step 2: Convert to DocumentElement objects
        from pipeline.claim_extractor import DocumentElement
        elements = [
            DocumentElement(
                element_id=e["element_id"],
                text=e["text"],
                page=e["page"],
                element_type=e["element_type"],
                section_path=e.get("section_path", []),
            )
            for e in raw_elements
        ]

        # Step 3: Index document for RAG
        print(f"[{report_id[:8]}] Building search index...")
        indexer = get_indexer()
        indexer.index_document(raw_elements)

        # Step 4: Extract claims
        print(f"[{report_id[:8]}] Extracting environmental claims...")
        extractor = get_claim_extractor()
        claims = extractor.extract_claims(elements)
        print(f"[{report_id[:8]}] Found {len(claims)} environmental claims")

        if not claims:
            # No claims found — return empty report
            report_data = {
                "run_id": report_id,
                "doc_name": file.filename,
                "total_claims": 0,
                "total_flagged": 0,
                "categories": [],
                "risk_heatmap": {},
                "verification_tasks": [],
                "message": "No environmental claims detected in this document.",
            }
            reports_store[report_id] = report_data
            return report_data

        # Step 5: Pre-analysis — BERTopic + table data + LLM Judge
        document_topics = []
        try:
            from bertopic import BERTopic
            all_texts = [e.text for e in elements if len(e.text) > 50]
            if len(all_texts) >= 5:
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

        table_data = [e for e in raw_elements if e.get("table_data")]
        judge = get_llm_judge()
        retriever = get_retriever()
        reg_retriever = get_regulatory_retriever()

        # Step 6: Run detection modules
        all_verdicts = []

        # Module 1: Vague Claims
        print(f"[{report_id[:8]}] Running Module 1: Vague Claims...")
        vague_mod = get_vague_module()
        vague_verdicts = vague_mod.analyze(claims, llm_judge=judge)
        all_verdicts.extend(vague_verdicts)
        print(f"[{report_id[:8]}]   → {sum(1 for v in vague_verdicts if v.verdict == 'flagged')} flagged")

        # Module 2: No Proof
        print(f"[{report_id[:8]}] Running Module 2: No Proof...")
        no_proof_mod = get_no_proof_module()
        no_proof_verdicts = no_proof_mod.analyze(
            claims,
            retriever=retriever,
            regulatory_retriever=reg_retriever,
            llm_judge=judge,
        )
        all_verdicts.extend(no_proof_verdicts)
        print(f"[{report_id[:8]}]   → {sum(1 for v in no_proof_verdicts if v.verdict == 'flagged')} flagged")

        # Module 3: Irrelevant Claims
        print(f"[{report_id[:8]}] Running Module 3: Irrelevant Claims...")
        irr_mod = get_irrelevant_module()
        irr_verdicts = irr_mod.analyze(
            claims,
            regulatory_retriever=reg_retriever,
            llm_judge=judge,
        )
        all_verdicts.extend(irr_verdicts)
        print(f"[{report_id[:8]}]   → {sum(1 for v in irr_verdicts if v.verdict == 'flagged')} flagged")

        # Module 4: Lesser of Two Evils
        print(f"[{report_id[:8]}] Running Module 4: Lesser of Two Evils...")
        lesser_mod = get_lesser_evil_module()
        lesser_verdicts = lesser_mod.analyze(claims, llm_judge=judge)
        all_verdicts.extend(lesser_verdicts)
        print(f"[{report_id[:8]}]   → {sum(1 for v in lesser_verdicts if v.verdict == 'flagged')} flagged")

        # Module 5: Hidden Tradeoffs
        print(f"[{report_id[:8]}] Running Module 5: Hidden Tradeoffs...")
        hidden_mod = get_hidden_tradeoffs_module()
        hidden_verdicts = hidden_mod.analyze(
            claims,
            retriever=retriever,
            document_topics=document_topics,
            llm_judge=judge,
        )
        all_verdicts.extend(hidden_verdicts)
        print(f"[{report_id[:8]}]   → {sum(1 for v in hidden_verdicts if v.verdict == 'flagged')} flagged")

        # Module 6: Fake Labels
        print(f"[{report_id[:8]}] Running Module 6: Fake Labels...")
        fake_mod = get_fake_labels_module()
        fake_verdicts = fake_mod.analyze(
            claims,
            regulatory_retriever=reg_retriever,
            llm_judge=judge,
        )
        all_verdicts.extend(fake_verdicts)
        print(f"[{report_id[:8]}]   → {sum(1 for v in fake_verdicts if v.verdict == 'flagged')} flagged")

        # Module 7: Fibbing
        print(f"[{report_id[:8]}] Running Module 7: Fibbing...")
        fib_mod = get_fibbing_module()
        fib_verdicts = fib_mod.analyze(
            claims,
            retriever=retriever,
            table_data=table_data,
            llm_judge=judge,
        )
        all_verdicts.extend(fib_verdicts)
        print(f"[{report_id[:8]}]   → {sum(1 for v in fib_verdicts if v.verdict == 'flagged')} flagged")

        # Step 7: Aggregate
        print(f"[{report_id[:8]}] Aggregating results...")
        aggregator = get_aggregator()
        report = aggregator.aggregate(all_verdicts, len(claims), file.filename)
        report_data = report.to_dict()

        # Store report
        reports_store[report_id] = report_data
        print(f"[{report_id[:8]}] Analysis complete. {report.total_flagged} total flags.")

        return report_data

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/analyze/text")
async def analyze_text(body: dict):
    """Analyze raw text (for testing without PDF).

    Body: {"text": "...", "doc_name": "test"}
    """
    text = body.get("text", "")
    doc_name = body.get("doc_name", "text_input")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text provided.")

    # Split into paragraph-sized elements
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 20]
    if not paragraphs:
        paragraphs = [text.strip()]

    from pipeline.claim_extractor import DocumentElement

    elements = [
        DocumentElement(
            element_id=f"text_{i}",
            text=para,
            page=1,
            element_type="NarrativeText",
            section_path=[],
        )
        for i, para in enumerate(paragraphs)
    ]

    # Index for RAG
    indexer = get_indexer()
    raw_elements = [{"element_id": e.element_id, "text": e.text, "page": e.page,
                      "element_type": e.element_type, "section_path": e.section_path}
                     for e in elements]
    indexer.index_document(raw_elements)

    # Extract claims
    extractor = get_claim_extractor()
    claims = extractor.extract_claims(elements)

    if not claims:
        return {"message": "No environmental claims detected.", "total_claims": 0}

    # Run modules
    all_verdicts = []
    judge = get_llm_judge()
    retriever = get_retriever()
    reg_retriever = get_regulatory_retriever()

    vague_mod = get_vague_module()
    all_verdicts.extend(vague_mod.analyze(claims, llm_judge=judge))

    no_proof_mod = get_no_proof_module()
    all_verdicts.extend(no_proof_mod.analyze(
        claims,
        retriever=retriever,
        regulatory_retriever=reg_retriever,
        llm_judge=judge,
    ))

    irr_mod = get_irrelevant_module()
    all_verdicts.extend(irr_mod.analyze(
        claims, regulatory_retriever=reg_retriever, llm_judge=judge,
    ))

    lesser_mod = get_lesser_evil_module()
    all_verdicts.extend(lesser_mod.analyze(claims, llm_judge=judge))

    hidden_mod = get_hidden_tradeoffs_module()
    all_verdicts.extend(hidden_mod.analyze(
        claims, retriever=retriever, llm_judge=judge,
    ))

    fake_mod = get_fake_labels_module()
    all_verdicts.extend(fake_mod.analyze(
        claims, regulatory_retriever=reg_retriever, llm_judge=judge,
    ))

    fib_mod = get_fibbing_module()
    all_verdicts.extend(fib_mod.analyze(
        claims, retriever=retriever, llm_judge=judge,
    ))

    # Aggregate
    aggregator = get_aggregator()
    report = aggregator.aggregate(all_verdicts, len(claims), doc_name)
    report_data = report.to_dict()

    report_id = report_data["run_id"]
    reports_store[report_id] = report_data

    return report_data


@app.get("/api/report/{report_id}")
async def get_report(report_id: str):
    """Get a previously generated report."""
    if report_id not in reports_store:
        raise HTTPException(status_code=404, detail="Report not found.")
    return reports_store[report_id]


@app.get("/api/report/{report_id}/category/{category_name}")
async def get_category(report_id: str, category_name: str):
    """Get items for a specific greenwashing category."""
    if report_id not in reports_store:
        raise HTTPException(status_code=404, detail="Report not found.")

    report = reports_store[report_id]
    for cat in report.get("categories", []):
        if cat["category"] == category_name:
            return cat

    raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found.")


@app.get("/api/report/{report_id}/summary")
async def get_summary(report_id: str):
    """Get summary with risk heatmap."""
    if report_id not in reports_store:
        raise HTTPException(status_code=404, detail="Report not found.")

    report = reports_store[report_id]
    return {
        "doc_name": report["doc_name"],
        "total_claims": report["total_claims"],
        "total_flagged": report["total_flagged"],
        "risk_heatmap": report["risk_heatmap"],
        "verification_tasks": report["verification_tasks"],
    }


@app.get("/api/reports")
async def list_reports():
    """List all available reports."""
    return [
        {
            "report_id": rid,
            "doc_name": r["doc_name"],
            "total_claims": r["total_claims"],
            "total_flagged": r["total_flagged"],
        }
        for rid, r in reports_store.items()
    ]

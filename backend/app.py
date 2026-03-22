"""Green Lens Backend — Greenwashing Detection API."""

import sys
import json
import uuid
import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import DATA_DIR

# ── Logging setup ─────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logging():
    """Configure logging to both console and rotating log files."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Main log file — rotates at 10 MB, keeps 5 backups
    main_handler = RotatingFileHandler(
        LOG_DIR / "green_lens.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8",
    )
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(fmt)
    root.addHandler(main_handler)

    # Error-only log file — easy to scan for failures
    error_handler = RotatingFileHandler(
        LOG_DIR / "errors.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)
    root.addHandler(error_handler)

setup_logging()
logger = logging.getLogger("green_lens")

app = FastAPI(title="Green Lens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for reports (MVP — no database)
reports_store: dict[str, dict] = {}

# Progress tracking for active analyses
progress_store: dict[str, dict] = {}


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


_llm_scorer = None

def get_llm_scorer():
    global _llm_scorer
    if _llm_scorer is None:
        from pipeline.llm.scorer import LLMScorer
        _llm_scorer = LLMScorer()
    return _llm_scorer


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


@app.get("/api/progress")
async def get_progress():
    """Return progress of the most recent active analysis."""
    if not progress_store:
        return {"active": False}
    latest_id = list(progress_store.keys())[-1]
    return {"active": True, **progress_store[latest_id]}


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

    filename = file.filename
    report_id = str(uuid.uuid4())
    log_prefix = report_id[:8]

    # Run the heavy pipeline in a thread so the event loop stays free
    # (this lets /api/progress respond while analysis runs)
    result = await asyncio.to_thread(
        _run_pipeline_sync, tmp_path, filename, report_id, log_prefix
    )
    return result


def _run_pipeline_sync(tmp_path: str, filename: str, report_id: str, log_prefix: str):
    """Synchronous pipeline — called via asyncio.to_thread()."""
    try:
        def update_progress(step: int, total: int, message: str):
            progress_store[report_id] = {
                "report_id": report_id,
                "step": step,
                "total_steps": total,
                "message": message,
                "percent": round(step / total * 100),
            }

        TOTAL_STEPS = 16

        # Step 1: Parse PDF
        update_progress(1, TOTAL_STEPS, "Parsing PDF...")
        logger.info("[%s] Parsing PDF: %s", log_prefix, filename)
        raw_elements = simple_text_extract(tmp_path)
        if not raw_elements or raw_elements[0].get("element_id") == "notice_0":
            raise RuntimeError("PDF parsing failed. Install pymupdf: pip install pymupdf")
        logger.info("[%s] Extracted %d text elements", log_prefix, len(raw_elements))

        # Step 2: Convert to DocumentElement objects
        from pipeline.claim_extractor import DocumentElement
        elements = [
            DocumentElement(
                element_id=e["element_id"],
                text=e["text"],
                page=e["page"],
                element_type=e["element_type"],
                section_path=e.get("section_path", []),
                element_role=e.get("element_role", "claim_candidate"),
            )
            for e in raw_elements
        ]

        # Step 3: Index document for RAG
        update_progress(2, TOTAL_STEPS, "Building search index...")
        logger.info("[%s] Building search index...", log_prefix)
        indexer = get_indexer()
        indexer.index_document(raw_elements)

        # Step 4: Extract claims
        update_progress(3, TOTAL_STEPS, "Extracting environmental claims...")
        logger.info("[%s] Extracting environmental claims...", log_prefix)
        extractor = get_claim_extractor()
        claims = extractor.extract_claims(elements, return_groups=True)
        total_sentence_claims = sum(
            len(c.claims) if hasattr(c, "claims") else 1 for c in claims
        )
        logger.info("[%s] Found %d claim groups (%d sentences)", log_prefix, len(claims), total_sentence_claims)

        if not claims:
            report_data = {
                "run_id": report_id,
                "doc_name": filename,
                "total_claims": 0,
                "total_flagged": 0,
                "categories": [],
                "risk_heatmap": {},
                "verification_tasks": [],
                "message": "No environmental claims detected in this document.",
            }
            reports_store[report_id] = report_data
            return report_data

        # Step 5: BERTopic
        update_progress(4, TOTAL_STEPS, "Running topic analysis (BERTopic)...")
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
                logger.info("[%s] BERTopic found %d topics", log_prefix, len(document_topics))
        except Exception as e:
            logger.warning("[%s] BERTopic skipped: %s", log_prefix, e)

        table_data = [e for e in raw_elements if e.get("table_data")]
        judge = get_llm_judge()
        retriever = get_retriever()
        reg_retriever = get_regulatory_retriever()

        # Step 6: Run detection modules
        all_verdicts = []

        modules = [
            (5, "Module 1/7: Analyzing Vague Claims...", "Vague Claims",
             lambda: get_vague_module().analyze(claims, llm_judge=judge)),
            (6, "Module 2/7: Checking for No Proof...", "No Proof",
             lambda: get_no_proof_module().analyze(claims, retriever=retriever, regulatory_retriever=reg_retriever, llm_judge=judge)),
            (7, "Module 3/7: Detecting Irrelevant Claims...", "Irrelevant Claims",
             lambda: get_irrelevant_module().analyze(claims, regulatory_retriever=reg_retriever, llm_judge=judge)),
            (8, "Module 4/7: Checking Lesser of Two Evils...", "Lesser of Two Evils",
             lambda: get_lesser_evil_module().analyze(claims, llm_judge=judge)),
            (9, "Module 5/7: Detecting Hidden Tradeoffs...", "Hidden Tradeoffs",
             lambda: get_hidden_tradeoffs_module().analyze(claims, retriever=retriever, document_topics=document_topics, llm_judge=judge)),
            (10, "Module 6/7: Checking Fake Labels...", "Fake Labels",
             lambda: get_fake_labels_module().analyze(claims, regulatory_retriever=reg_retriever, llm_judge=judge)),
            (11, "Module 7/7: Detecting Fibbing...", "Fibbing",
             lambda: get_fibbing_module().analyze(claims, retriever=retriever, table_data=table_data, llm_judge=judge)),
        ]

        for step_num, progress_msg, module_name, run_fn in modules:
            update_progress(step_num, TOTAL_STEPS, progress_msg)
            logger.info("[%s] Running %s...", log_prefix, module_name)
            try:
                verdicts = run_fn()
                all_verdicts.extend(verdicts)
                flagged = sum(1 for v in verdicts if v.verdict == "flagged")
                logger.info("[%s]   → %d flagged", log_prefix, flagged)
            except Exception as e:
                logger.error("[%s] %s module failed: %s", log_prefix, module_name, e, exc_info=True)

        # Step 7: Aggregate
        update_progress(12, TOTAL_STEPS, "Aggregating module results...")
        logger.info("[%s] Aggregating results...", log_prefix)
        aggregator = get_aggregator()
        report = aggregator.aggregate(all_verdicts, len(claims), filename)
        report_data = report.to_dict()

        # Enrich report items with ClaimGroup constituent info
        for cat in report_data.get("categories", []):
            for item in cat.get("items", []):
                for c in claims:
                    if hasattr(c, "claims") and c.claim_text == item.get("claim_text"):
                        item["constituent_sentences"] = [
                            {"text": sc.claim_text, "confidence": sc.confidence,
                             "offset": sc.sentence_offset}
                            for sc in c.claims
                        ]
                        item["representative_sentence"] = c.representative_sentence
                        break

        # Step 8: Stage 2 Sin Scoring
        update_progress(13, TOTAL_STEPS, "Stage 2: Scoring claims (LLM)...")
        logger.info("[%s] Stage 2: Scoring claims against 7 Sins...", log_prefix)
        scorer = get_llm_scorer()
        claim_judgments: dict[str, dict[str, dict]] = {}
        for cat in report_data.get("categories", []):
            module_name = cat.get("category", "")
            for item in cat.get("items", []):
                ct = item.get("claim_text", "")
                if ct not in claim_judgments:
                    claim_judgments[ct] = {}
                judgment = item.get("judgment")
                if judgment:
                    claim_judgments[ct][module_name] = judgment

        scored_count = 0
        claim_scores: dict[str, dict] = {}
        for ct, judgments in claim_judgments.items():
            try:
                result = scorer.score_claim(ct, judgments)
                if result is not None:
                    claim_scores[ct] = result.to_dict()
                    scored_count += 1
                    logger.info("[%s]   Scored claim (%d/%d): risk=%s",
                                log_prefix, scored_count, len(claim_judgments), result.claim_risk)
            except Exception:
                logger.error("[%s] Scoring failed for claim: %s...", log_prefix, ct[:80], exc_info=True)
                continue

        # Attach scores to verdict items
        for cat in report_data.get("categories", []):
            for item in cat.get("items", []):
                ct = item.get("claim_text", "")
                if ct in claim_scores:
                    item["sin_scores"] = claim_scores[ct]["sin_scores"]
                    item["signal_breakdowns"] = claim_scores[ct]["signal_breakdowns"]
                    item["claim_risk"] = claim_scores[ct]["claim_risk"]
                    item["top_drivers"] = claim_scores[ct]["top_drivers"]

        update_progress(15, TOTAL_STEPS, f"Scoring complete ({scored_count}/{len(claim_judgments)} claims).")
        logger.info("[%s] Scoring complete. %d/%d claims scored.", log_prefix, scored_count, len(claim_judgments))

        # Compute scoring summary
        if claim_scores:
            sin_names = [
                "hidden_tradeoff", "no_proof", "vagueness", "false_labels",
                "irrelevance", "lesser_of_two_evils", "fibbing",
            ]
            n_scored = len(claim_scores)
            sin_averages = {}
            for sin in sin_names:
                total = sum(cs["sin_scores"].get(sin, 0) for cs in claim_scores.values())
                sin_averages[sin] = round(total / n_scored, 1)
            avg_risk = round(
                sum(cs["claim_risk"] for cs in claim_scores.values()) / n_scored, 1
            )
            report_data["scoring_summary"] = {
                "total_claims_scored": n_scored,
                "average_claim_risk": avg_risk,
                "average_sin_scores": sin_averages,
            }
            logger.info("[%s] Average claim risk: %s/100 | Top sin: %s (%s/100)",
                        log_prefix, avg_risk,
                        max(sin_averages, key=sin_averages.get),
                        max(sin_averages.values()))

        # Store report
        reports_store[report_id] = report_data
        update_progress(16, TOTAL_STEPS, "Analysis complete!")
        logger.info("[%s] Analysis complete. %d total flags.", log_prefix, report.total_flagged)
        progress_store.pop(report_id, None)

        return report_data

    except Exception as e:
        logger.error("[%s] Analysis failed: %s", log_prefix, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
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

    # Extract claims (grouped)
    extractor = get_claim_extractor()
    claims = extractor.extract_claims(elements, return_groups=True)

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

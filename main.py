"""Genesis Portfolio Pulse — FastAPI backend."""
import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import rag
import db

load_dotenv()

ADMIN_KEY = os.getenv("ADMIN_KEY", "")

app = FastAPI(
    title="Genesis Portfolio Pulse",
    description="AI-powered intelligence dashboard for Genesis portfolio companies",
    version="0.1.0",
)

# Allow Lovable frontend to call this API from anywhere during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load companies once at startup
COMPANIES_PATH = Path(__file__).parent / "companies.json"
with open(COMPANIES_PATH) as f:
    COMPANIES_DATA = json.load(f)
COMPANIES = COMPANIES_DATA["companies"]
COMPANIES_BY_ID = {c["id"]: c for c in COMPANIES}

# Initialize DB on startup
db.init_db()


# ────────────────────────────────────────────────────────────
# Public endpoints
# ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Genesis Portfolio Pulse",
        "status": "ok",
        "company_count": len(COMPANIES),
    }


@app.get("/companies")
def list_companies(active_only: bool = Query(False, description="Only return companies with recent signals")):
    """Return tracked Genesis portfolio companies. Set active_only=true to filter to those with signals."""
    if not active_only:
        return {"count": len(COMPANIES), "companies": COMPANIES}

    with db.get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT company_id FROM signals").fetchall()
        active_ids = {r["company_id"] for r in rows}

    active_companies = [c for c in COMPANIES if c["id"] in active_ids]
    return {"count": len(active_companies), "companies": active_companies}


@app.get("/companies/{company_id}")
def get_company(company_id: str):
    """Return one company by ID."""
    company = COMPANIES_BY_ID.get(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company


@app.get("/signals")
def list_signals(
    company_id: str = Query(None),
    signal_type: str = Query(None),
    limit: int = Query(100, le=500),
):
    """Return cached signals, optionally filtered by company or type."""
    signals = db.get_signals(company_id=company_id, signal_type=signal_type, limit=limit)
    for s in signals:
        company = COMPANIES_BY_ID.get(s["company_id"])
        s["company_name"] = company["name"] if company else s["company_id"]
        s["company_sector"] = company["sector"] if company else None
    return {"count": len(signals), "signals": signals}


@app.get("/digest")
def get_digest():
    """Return the latest weekly digest."""
    digest = db.get_latest_digest()
    if not digest:
        return {"content": None, "message": "No digest generated yet. Run ingest.py first."}
    return digest


@app.get("/trends")
def get_trends():
    """Return latest cross-portfolio trend analysis."""
    trends = db.get_latest_trends()
    if not trends:
        return {"content": None, "message": "No trends generated yet."}
    return trends


@app.get("/ask")
def ask(q: str = Query(..., min_length=3, description="Natural-language question")):
    """RAG endpoint: answer questions about the portfolio with citations."""
    return rag.query(q)


# ────────────────────────────────────────────────────────────
# Admin endpoints (protected by ADMIN_KEY)
# ────────────────────────────────────────────────────────────

def _check_admin(provided_key: str):
    if not ADMIN_KEY or provided_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.post("/admin/ingest")
def admin_ingest(key: str = Query(...)):
    """Run full Firecrawl ingestion. Protected by ADMIN_KEY env var."""
    _check_admin(key)
    import ingest
    ingest.ingest_all()
    return {"status": "ok", "step": "ingest_complete"}


@app.post("/admin/analyze")
def admin_analyze(key: str = Query(...)):
    """Run AI analysis: classification + digest + trends."""
    _check_admin(key)
    import analyze
    analyze.run_full_analysis()
    return {"status": "ok", "step": "analysis_complete"}


@app.post("/admin/reindex")
def admin_reindex(key: str = Query(...)):
    """Rebuild ChromaDB index from current signals."""
    _check_admin(key)
    rag.index_all_signals()
    return {"status": "ok", "step": "reindex_complete"}


@app.post("/admin/refresh-all")
def admin_refresh_all(key: str = Query(...)):
    """One-shot: ingest + analyze + reindex. Use this for full data refresh."""
    _check_admin(key)
    import ingest
    import analyze
    ingest.ingest_all()
    analyze.run_full_analysis()
    rag.index_all_signals()
    return {"status": "ok", "step": "full_refresh_complete"}
"""Genesis Portfolio Pulse — FastAPI backend."""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import rag
import db

load_dotenv()

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

    # Get distinct company_ids that have at least one signal
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
    # Enrich with company name
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
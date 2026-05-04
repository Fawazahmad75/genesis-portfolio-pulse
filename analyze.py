"""AI analysis layer: classify signals and generate weekly digest using Groq."""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

import db

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing from .env")

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"  # Groq's strongest general model

COMPANIES_PATH = Path(__file__).parent / "companies.json"
with open(COMPANIES_PATH) as f:
    COMPANIES = {c["id"]: c for c in json.load(f)["companies"]}


def _llm(prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> str:
    """Single helper for all Groq calls."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ────────────────────────────────────────────────────────────
# 1. Signal Classification
# ────────────────────────────────────────────────────────────
def classify_signals():
    signals = db.get_signals(limit=500)
    if not signals:
        print("No signals to classify.")
        return

    signal_blocks = []
    for s in signals:
        company = COMPANIES.get(s["company_id"], {})
        signal_blocks.append(
            f"ID: {s['id']}\n"
            f"Company: {company.get('name', s['company_id'])} ({company.get('sector', 'unknown')})\n"
            f"Title: {s['title']}\n"
            f"Snippet: {(s.get('snippet') or '')[:300]}\n"
        )

    prompt = f"""You are analyzing news signals for a startup incubator's portfolio dashboard. For each signal below, classify:

1. signal_type — one of: news, funding, hiring, product, partnership, regulatory, press
2. importance — one of: high, medium, low

High importance = funding rounds, major product launches, acquisitions, regulatory milestones, named partnerships with notable companies, significant hiring (5+ roles).
Medium importance = standard product updates, single hires, mentions in industry roundups.
Low importance = passing mentions, generic press coverage, content marketing posts.

Return ONLY a JSON array. No prose, no markdown fences. Format:
[{{"id": <int>, "signal_type": "<str>", "importance": "<str>"}}, ...]

Signals to classify:

{chr(10).join(signal_blocks)}
"""

    print(f"Classifying {len(signals)} signals via Groq...")
    raw = _llm(prompt, max_tokens=4000, temperature=0.1)

    # Strip code fences if model added them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        classifications = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse JSON: {e}")
        print(f"  Raw (first 500 chars): {raw[:500]}")
        return

    with db.get_conn() as conn:
        for c in classifications:
            conn.execute(
                "UPDATE signals SET signal_type = ?, importance = ? WHERE id = ?",
                (c.get("signal_type", "news"), c.get("importance", "medium"), c["id"]),
            )
    print(f"✓ Classified {len(classifications)} signals.")


# ────────────────────────────────────────────────────────────
# 2. Weekly Digest
# ────────────────────────────────────────────────────────────
def generate_digest():
    signals = db.get_signals(limit=500)
    if not signals:
        print("No signals to summarize.")
        return

    relevant = [s for s in signals if s.get("importance") in ("high", "medium")]
    if not relevant:
        relevant = signals

    signal_blocks = []
    for s in relevant:
        company = COMPANIES.get(s["company_id"], {})
        signal_blocks.append(
            f"- {company.get('name', s['company_id'])} "
            f"[{s.get('importance')}, {s.get('signal_type')}]: {s['title']}"
        )

    prompt = f"""You are writing the weekly executive digest for Genesis, a startup incubator in Newfoundland & Labrador. Below are recent news and activity signals across the active portfolio companies.

Write a 200-word executive summary highlighting the 3-5 most important developments this week. Be concrete: name companies, name what happened, name why it matters. Group thematically where natural (e.g., "On the healthcare front..."). Use markdown formatting with bold company names. Skip filler. Skip closing remarks.

Recent signals ({len(relevant)} total):

{chr(10).join(signal_blocks)}
"""

    print("Generating digest via Groq...")
    digest_content = _llm(prompt, max_tokens=1000, temperature=0.5)
    db.save_digest(digest_content, signal_count=len(signals))
    print(f"✓ Digest saved ({len(digest_content)} chars).")


# ────────────────────────────────────────────────────────────
# 3. Trend Analysis
# ────────────────────────────────────────────────────────────
def generate_trends():
    signals = db.get_signals(limit=500)
    if not signals:
        print("No signals for trends.")
        return

    by_sector = {}
    for s in signals:
        company = COMPANIES.get(s["company_id"], {})
        sector = company.get("sector", "Other")
        by_sector.setdefault(sector, []).append(
            f"{company.get('name', s['company_id'])}: {s['title']}"
        )

    sector_blocks = []
    for sector, items in by_sector.items():
        sector_blocks.append(f"\n## {sector} ({len(items)} signals)\n" + "\n".join(f"- {i}" for i in items))

    prompt = f"""You are analyzing the Genesis portfolio for cross-cutting trends. Below are recent signals grouped by sector.

Identify 2-4 notable patterns or trends. Examples:
- "Healthcare cluster shows momentum: 3 companies announced milestones"
- "AI/ML adoption accelerating across non-tech sectors"
- "Hiring concentrated in [sector]"

Be concrete and brief. Use bullet points with bold lead-ins. Skip generic observations. Only call out real patterns visible in the data — if data is too sparse for a real pattern, say so honestly.

Signals by sector:
{chr(10).join(sector_blocks)}
"""

    print("Generating trends via Groq...")
    trends_content = _llm(prompt, max_tokens=800, temperature=0.4)
    db.save_trends(trends_content)
    print(f"✓ Trends saved ({len(trends_content)} chars).")


def run_full_analysis():
    print("=" * 60)
    print("GENESIS PORTFOLIO PULSE — AI Analysis (Groq)")
    print("=" * 60)
    classify_signals()
    print()
    generate_digest()
    print()
    generate_trends()
    print()
    print("✓ Analysis complete.")


if __name__ == "__main__":
    run_full_analysis()
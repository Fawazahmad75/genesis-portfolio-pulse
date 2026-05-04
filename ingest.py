"""
Genesis Portfolio Pulse — Firecrawl ingestion with rate-limit handling.
"""
import os
import re
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from firecrawl import Firecrawl

import db

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not FIRECRAWL_API_KEY:
    raise RuntimeError("FIRECRAWL_API_KEY missing from .env")

firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)

COMPANIES_PATH = Path(__file__).parent / "companies.json"
with open(COMPANIES_PATH) as f:
    COMPANIES = json.load(f)["companies"]


CONTENT_KEYWORDS = [
    "blog", "news", "press", "announcement", "article", "post",
    "update", "release", "story", "stories", "insight", "insights",
    "case-stud", "casestud", "media", "newsroom", "publication",
    "forecast",
]

SKIP_KEYWORDS = [
    "/tag/", "/category/", "/author/", "/page/", "/feed",
    "/login", "/signup", "/contact", "/about", "/privacy",
    "/terms", "/cookie", "/careers", "/jobs", "/team",
    ".pdf", ".jpg", ".png", ".css", ".js",
]


def looks_like_content(url: str) -> bool:
    url_lower = url.lower()
    if any(skip in url_lower for skip in SKIP_KEYWORDS):
        return False
    return any(kw in url_lower for kw in CONTENT_KEYWORDS)


def _retry_on_rate_limit(fn, *args, max_retries=3, **kwargs):
    """Run a Firecrawl call; if rate-limited, sleep until reset and retry."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if "Rate Limit Exceeded" in err or "rate limit" in err.lower():
                m = re.search(r"retry after (\d+)s", err)
                wait = int(m.group(1)) + 2 if m else 30
                print(f"  ⏸  rate limited, sleeping {wait}s...", end=" ", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Max retries ({max_retries}) exceeded for {fn.__name__}")


def map_company_site(website: str, max_urls: int = 30) -> list[str]:
    try:
        result = _retry_on_rate_limit(firecrawl.map, website, limit=100)
        if hasattr(result, "links"):
            links = [lk.url if hasattr(lk, "url") else str(lk) for lk in result.links]
        else:
            links = []
    except Exception as e:
        print(f"  ! map error: {str(e)[:80]}", end=" ")
        return []

    content_urls = [u for u in links if looks_like_content(u)]
    return content_urls[:max_urls]


def scrape_url(url: str) -> dict | None:
    try:
        doc = _retry_on_rate_limit(
            firecrawl.scrape,
            url,
            formats=["markdown"],
            only_main_content=True,
        )
        markdown = getattr(doc, "markdown", "") or ""
        metadata = getattr(doc, "metadata", None)

        if not markdown or len(markdown.strip()) < 100:
            return None

        def meta_get(key, default=None):
            if metadata is None:
                return default
            if isinstance(metadata, dict):
                return metadata.get(key, default)
            return getattr(metadata, key, default)

        return {
            "title": meta_get("title") or meta_get("og_title") or meta_get("ogTitle") or "(untitled)",
            "url": url,
            "content": markdown[:2000],
            "published_date": meta_get("published_time") or meta_get("article_published_time"),
            "description": meta_get("description") or meta_get("og_description", "") or "",
        }
    except Exception as e:
        print(f"  ! scrape error: {str(e)[:60]}", end=" ")
        return None


def ingest_company(company: dict, max_pages: int = 3) -> int:
    name = company["name"]
    website = company.get("website")
    if not website:
        return 0

    content_urls = map_company_site(website)
    if not content_urls:
        return 0

    inserted = 0
    for url in content_urls[:max_pages]:
        signal = scrape_url(url)
        if not signal:
            continue
        domain = website.replace("https://", "").replace("http://", "").split("/")[0]
        db.insert_signal(
            company_id=company["id"],
            title=signal["title"],
            url=signal["url"],
            source=domain,
            published_date=signal["published_date"],
            snippet=signal["content"][:500],
            signal_type="news",
            importance="medium",
        )
        inserted += 1
        time.sleep(11)  # ~6 req/min ceiling, with margin

    return inserted


def ingest_all():
    db.init_db()
    print(f"Ingesting from {len(COMPANIES)} company websites via Firecrawl...\n")

    total = 0
    for i, company in enumerate(COMPANIES, start=1):
        print(f"[{i}/{len(COMPANIES)}] {company['name']}...", end=" ", flush=True)
        if not company.get("website"):
            print("no website, skip")
            continue
        count = ingest_company(company)
        total += count
        print(f"{count} signals")
        time.sleep(11)  # space out per-company map calls too

    print(f"\n✓ Ingestion complete. {total} signals across {len(COMPANIES)} companies.")


if __name__ == "__main__":
    ingest_all()
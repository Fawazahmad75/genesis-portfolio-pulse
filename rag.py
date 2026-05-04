"""ChromaDB RAG layer for natural-language portfolio queries."""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from groq import Groq
from chromadb.utils import embedding_functions

import db

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
LLM_MODEL = "llama-3.3-70b-versatile"

CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "portfolio_signals"

COMPANIES_PATH = Path(__file__).parent / "companies.json"
with open(COMPANIES_PATH) as f:
    COMPANIES = {c["id"]: c for c in json.load(f)["companies"]}


# Lazy-init Chroma so the import is fast
_client = None
_collection = None

def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embed_fn,
            )
    return _collection


def index_all_signals():
    """Embed every signal in ChromaDB. Idempotent — safe to re-run."""
    collection = _get_collection()
    signals = db.get_signals(limit=1000)
    if not signals:
        print("No signals to index.")
        return

    # Wipe and rebuild — keeps it simple and correct
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    docs, metadatas, ids = [], [], []
    for s in signals:
        company = COMPANIES.get(s["company_id"], {})
        company_name = company.get("name", s["company_id"])

        # The embedded text combines title + snippet for richer matching
        text = f"{company_name}: {s['title']}\n\n{s.get('snippet') or ''}"
        docs.append(text)
        metadatas.append({
            "signal_id": s["id"],
            "company_id": s["company_id"],
            "company_name": company_name,
            "company_sector": company.get("sector", ""),
            "title": s["title"],
            "url": s["url"],
            "signal_type": s.get("signal_type", "news"),
            "importance": s.get("importance", "medium"),
            "published_date": s.get("published_date") or "",
        })
        ids.append(f"signal_{s['id']}")

    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    print(f"✓ Indexed {len(docs)} signals into ChromaDB.")


def query(question: str, k: int = 5) -> dict:
    """Run a RAG query: retrieve top-k signals, generate cited answer."""
    collection = _get_collection()
    results = collection.query(query_texts=[question], n_results=k)

    if not results["documents"] or not results["documents"][0]:
        return {
            "answer": "I couldn't find any signals relevant to that question in the portfolio data.",
            "sources": [],
        }

    # Build context from retrieved signals
    retrieved = []
    context_blocks = []
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
        retrieved.append(meta)
        context_blocks.append(
            f"[{i}] {meta['company_name']} ({meta['company_sector']}): "
            f"{meta['title']}\n{doc[:400]}"
        )

    prompt = f"""You are answering questions about Genesis incubator's portfolio companies based on retrieved signals.

Question: {question}

Retrieved signals (numbered for citation):
{chr(10).join(context_blocks)}

Instructions:
- Answer the question using ONLY the retrieved signals above
- Cite sources inline using [1], [2], etc. matching the signal numbers
- If the signals don't contain enough information to answer, say so honestly
- Be concise (3-5 sentences)
- Use markdown formatting and **bold company names**
"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.3,
    )
    answer = response.choices[0].message.content.strip()

    # Build clean sources list for the frontend
    sources = []
    for i, meta in enumerate(retrieved, start=1):
        sources.append({
            "index": i,
            "company_name": meta["company_name"],
            "company_sector": meta["company_sector"],
            "title": meta["title"],
            "url": meta["url"],
            "signal_type": meta["signal_type"],
        })

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    index_all_signals()
    # Quick sanity test
    print("\n--- Test query: 'What is happening in healthcare?' ---")
    result = query("What is happening in healthcare?")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['company_name']}: {s['title']}")
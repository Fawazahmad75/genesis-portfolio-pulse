"""SQLite cache layer for portfolio signals and digests."""
import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "pulse.db"


@contextmanager
def get_conn():
    """Yields a SQLite connection with row factory set to dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                source TEXT,
                published_date TEXT,
                snippet TEXT,
                signal_type TEXT DEFAULT 'news',
                importance TEXT DEFAULT 'medium',
                fetched_at TEXT NOT NULL,
                UNIQUE(company_id, url)
            );

            CREATE TABLE IF NOT EXISTS digests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                signal_count INTEGER
            );

            CREATE TABLE IF NOT EXISTS trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                generated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_signals_company ON signals(company_id);
            CREATE INDEX IF NOT EXISTS idx_signals_fetched ON signals(fetched_at);
        """)


def insert_signal(company_id: str, title: str, url: str, source: str = None,
                  published_date: str = None, snippet: str = None,
                  signal_type: str = "news", importance: str = "medium"):
    """Insert a signal; ignore if (company_id, url) already exists."""
    with get_conn() as conn:
        try:
            conn.execute("""
                INSERT INTO signals
                (company_id, title, url, source, published_date, snippet,
                 signal_type, importance, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (company_id, title, url, source, published_date, snippet,
                  signal_type, importance, datetime.now(timezone.utc).isoformat()))
        except sqlite3.IntegrityError:
            pass  # Duplicate — already cached


def get_signals(company_id: str = None, signal_type: str = None, limit: int = 100):
    """Fetch signals, optionally filtered."""
    query = "SELECT * FROM signals WHERE 1=1"
    params = []
    if company_id:
        query += " AND company_id = ?"
        params.append(company_id)
    if signal_type:
        query += " AND signal_type = ?"
        params.append(signal_type)
    query += " ORDER BY published_date DESC, fetched_at DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def save_digest(content: str, signal_count: int):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO digests (content, generated_at, signal_count)
            VALUES (?, ?, ?)
        """, (content, datetime.now(timezone.utc).isoformat(), signal_count))


def get_latest_digest():
    with get_conn() as conn:
        row = conn.execute("""
            SELECT * FROM digests ORDER BY generated_at DESC LIMIT 1
        """).fetchone()
        return dict(row) if row else None


def save_trends(content: str):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO trends (content, generated_at)
            VALUES (?, ?)
        """, (content, datetime.now(timezone.utc).isoformat()))


def get_latest_trends():
    with get_conn() as conn:
        row = conn.execute("""
            SELECT * FROM trends ORDER BY generated_at DESC LIMIT 1
        """).fetchone()
        return dict(row) if row else None


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
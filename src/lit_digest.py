from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List



ARXIV_API = "http://export.arxiv.org/api/query"
DEFAULT_RSS_FEEDS = {
    "nature": "https://www.nature.com/nature.rss",
    "science": "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science",
    "prl": "https://journals.aps.org/rss/recent/prl.xml",
}

CATEGORY_RULES = {
    "quantum": ["quantum", "qubit", "trapped ion", "entanglement", "rydberg"],
    "ml-ai": ["machine learning", "deep learning", "transformer", "llm", "neural"],
    "materials": ["superconduct", "semiconductor", "graphene", "battery", "catalyst"],
    "bio": ["protein", "genome", "cell", "rna", "drug", "clinical"],
    "astro": ["galaxy", "cosmology", "exoplanet", "black hole", "gravitational"],
}


@dataclass
class Paper:
    source: str
    source_id: str
    title: str
    summary: str
    authors: list[str]
    link: str
    published_at: str


class PreferenceProfile:
    def __init__(self, keywords: dict[str, float], authors: dict[str, float]):
        self.keywords = {k.lower(): v for k, v in keywords.items()}
        self.authors = {a.lower(): v for a, v in authors.items()}

    @classmethod
    def from_history(cls, history_path: str) -> "PreferenceProfile":
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        keywords: dict[str, float] = {}
        authors: dict[str, float] = {}

        for item in data.get("likes", []):
            weight = float(item.get("weight", 1.0))
            for kw in item.get("keywords", []):
                k = kw.strip().lower()
                if k:
                    keywords[k] = keywords.get(k, 0.0) + weight
            for author in item.get("authors", []):
                a = author.strip().lower()
                if a:
                    authors[a] = authors.get(a, 0.0) + weight

        return cls(keywords=keywords, authors=authors)

    def score(self, paper: Paper) -> float:
        text = f"{paper.title} {paper.summary}".lower()
        score = 0.0

        for kw, w in self.keywords.items():
            if kw in text:
                score += w

        for author in paper.authors:
            aw = self.authors.get(author.lower(), 0.0)
            score += aw * 1.5

        novelty_penalty = max(0, len(text.split()) - 800) / 400
        return round(score - novelty_penalty, 3)


class PaperStore:
    def __init__(self, db_path: str = "papers.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT,
                    authors TEXT,
                    link TEXT,
                    published_at TEXT,
                    category TEXT,
                    score REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE(source, source_id)
                )
                """
            )

    def upsert(self, paper: Paper, category: str, score: float) -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO papers (source, source_id, title, summary, authors, link, published_at, category, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, source_id) DO UPDATE SET
                    title=excluded.title,
                    summary=excluded.summary,
                    authors=excluded.authors,
                    link=excluded.link,
                    published_at=excluded.published_at,
                    category=excluded.category,
                    score=excluded.score
                """,
                (
                    paper.source,
                    paper.source_id,
                    paper.title,
                    paper.summary,
                    json.dumps(paper.authors, ensure_ascii=False),
                    paper.link,
                    paper.published_at,
                    category,
                    score,
                    now,
                ),
            )

    def top_papers(self, limit: int = 20) -> list[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT source, title, authors, link, published_at, category, score
                FROM papers
                ORDER BY score DESC, published_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return rows


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def detect_category(paper: Paper) -> str:
    text = f"{paper.title} {paper.summary}".lower()
    scores = {}
    for cat, keys in CATEGORY_RULES.items():
        scores[cat] = sum(1 for kw in keys if kw in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def fetch_arxiv(query: str = "all:quantum", max_results: int = 40) -> list[Paper]:
    params = {"search_query": query, "start": 0, "max_results": max_results}
    import requests
    import feedparser

    res = requests.get(ARXIV_API, params=params, timeout=30)
    res.raise_for_status()

    feed = feedparser.parse(res.text)
    papers = []
    for entry in feed.entries:
        papers.append(
            Paper(
                source="arxiv",
                source_id=entry.get("id", ""),
                title=clean_text(entry.get("title", "")),
                summary=clean_text(entry.get("summary", "")),
                authors=[a.get("name", "") for a in entry.get("authors", [])],
                link=entry.get("link", ""),
                published_at=_parse_date(entry.get("published", "")),
            )
        )
    return papers


def fetch_rss_feeds(feeds: dict[str, str] | None = None) -> list[Paper]:
    import feedparser

    feeds = feeds or DEFAULT_RSS_FEEDS
    papers: list[Paper] = []

    for source, url in feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            authors = []
            if "authors" in entry:
                authors = [a.get("name", "") for a in entry.authors]
            elif "author" in entry:
                authors = [entry.get("author", "")]

            papers.append(
                Paper(
                    source=source,
                    source_id=entry.get("id") or entry.get("link", ""),
                    title=clean_text(entry.get("title", "")),
                    summary=clean_text(entry.get("summary", "")),
                    authors=authors,
                    link=entry.get("link", ""),
                    published_at=_parse_date(entry.get("published", "") or entry.get("updated", "")),
                )
            )
    return papers


def _parse_date(raw: str) -> str:
    if not raw:
        return ""
    try:
        from dateutil import parser as date_parser
        return date_parser.parse(raw).isoformat()
    except Exception:
        return raw


def build_digest_markdown(rows: Iterable[tuple]) -> str:
    lines = ["# Personalized Paper Digest", ""]
    for source, title, authors_json, link, published, category, score in rows:
        try:
            authors = ", ".join(json.loads(authors_json))
        except Exception:
            authors = authors_json
        lines.extend(
            [
                f"## [{title}]({link})",
                f"- Source: {source}",
                f"- Authors: {authors}",
                f"- Published: {published}",
                f"- Category: {category}",
                f"- Relevance score: {score}",
                "",
            ]
        )
    return "\n".join(lines)


def sync_to_zotero(rows: Iterable[tuple], top_n: int = 10) -> None:
    api_key = os.getenv("ZOTERO_API_KEY")
    user_id = os.getenv("ZOTERO_USER_ID")
    collection_key = os.getenv("ZOTERO_COLLECTION_KEY", "")
    if not api_key or not user_id:
        print("[zotero] missing ZOTERO_API_KEY/ZOTERO_USER_ID; skip sync")
        return

    url = f"https://api.zotero.org/users/{user_id}/items"
    headers = {
        "Zotero-API-Key": api_key,
        "Content-Type": "application/json",
    }

    items = []
    for i, (_source, title, authors_json, link, published, category, score) in enumerate(rows):
        if i >= top_n:
            break
        try:
            author_names = json.loads(authors_json)
        except Exception:
            author_names = [authors_json]

        creators = []
        for name in author_names:
            parts = name.split(" ", 1)
            if len(parts) == 2:
                creators.append({"creatorType": "author", "firstName": parts[0], "lastName": parts[1]})
            else:
                creators.append({"creatorType": "author", "name": name})

        item = {
            "itemType": "journalArticle",
            "title": title,
            "url": link,
            "date": published[:10] if published else "",
            "creators": creators,
            "tags": [{"tag": f"auto:{category}"}, {"tag": f"score:{score}"}],
            "abstractNote": "Added by lit-digest auto screening pipeline.",
        }
        if collection_key:
            item["collections"] = [collection_key]
        items.append(item)

    if not items:
        return

    import requests

    resp = requests.post(url, headers=headers, data=json.dumps(items), timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Zotero sync failed: {resp.status_code} {resp.text}")



def recluster_existing(db_path: str = "papers.db") -> None:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, source, source_id, title, summary, authors, link, published_at FROM papers").fetchall()
        for row in rows:
            pid, source, source_id, title, summary, authors_json, link, published = row
            try:
                authors = json.loads(authors_json)
            except Exception:
                authors = []
            paper = Paper(source, source_id, title, summary, authors, link, published)
            cat = detect_category(paper)
            conn.execute("UPDATE papers SET category=? WHERE id=?", (cat, pid))


def run_pipeline(history_path: str = "preferences.json", db_path: str = "papers.db") -> None:
    profile = PreferenceProfile.from_history(history_path)
    store = PaperStore(db_path)

    papers = fetch_arxiv() + fetch_rss_feeds()
    for p in papers:
        category = detect_category(p)
        score = profile.score(p)
        store.upsert(p, category=category, score=score)

    top_rows = store.top_papers(limit=30)
    digest = build_digest_markdown(top_rows)
    with open("digest.md", "w", encoding="utf-8") as f:
        f.write(digest)

    sync_to_zotero(top_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Literature subscription and recommendation pipeline")
    parser.add_argument("--history", default="preferences.json", help="Preference history JSON")
    parser.add_argument("--db", default="papers.db", help="SQLite path")
    parser.add_argument("--recluster", action="store_true", help="Re-classify existing papers")
    args = parser.parse_args()

    if args.recluster:
        recluster_existing(args.db)
    else:
        run_pipeline(history_path=args.history, db_path=args.db)


if __name__ == "__main__":
    main()

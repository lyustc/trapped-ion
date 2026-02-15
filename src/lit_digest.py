from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

ARXIV_API = "http://export.arxiv.org/api/query"
EMBED_DIM = 256

DEFAULT_RSS_FEEDS = {
    "nature": "https://www.nature.com/nature.rss",
    "nature-physics": "https://www.nature.com/nphys.rss",
    "nature-photonics": "https://www.nature.com/nphoton.rss",
    "nature-communications": "https://www.nature.com/ncomms.rss",
    "science": "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science",
    "science-advances": "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=sciadv",
    "prl": "https://journals.aps.org/rss/recent/prl.xml",
    "pr-applied": "https://journals.aps.org/rss/recent/prapplied.xml",
    "prx": "https://journals.aps.org/rss/recent/prx.xml",
    "prx-quantum": "https://journals.aps.org/rss/recent/prxquantum.xml",
}

CATEGORY_RULES = {
    "quantum": ["quantum", "qubit", "trapped ion", "entanglement", "rydberg"],
    "ml-ai": ["machine learning", "deep learning", "transformer", "llm", "neural"],
    "materials": ["superconduct", "semiconductor", "graphene", "battery", "catalyst"],
    "bio": ["protein", "genome", "cell", "rna", "drug", "clinical"],
    "astro": ["galaxy", "cosmology", "exoplanet", "black hole", "gravitational"],
}

TOPIC_TAG_RULES = {
    "quantum-computing": ["quantum computing", "fault tolerant", "quantum error correction", "quantum gate", "qubit"],
    "quantum-simulation": ["quantum simulation", "simulator", "hamiltonian", "many-body"],
    "quantum-sensing": ["quantum sensing", "magnetometer", "interferometer", "metrology"],
    "theory": ["theory", "theoretical", "proof", "model", "bounds", "derivation"],
    "experiment": ["experiment", "experimental", "measured", "demonstrate", "measurement", "device"],
}

ARTICLE_TYPE_RULES = {
    "review": ["review", "survey", "perspective", "overview"],
    "news": ["news", "editorial", "comment", "opinion", "research highlight", "news & views"],
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
        self.keywords = {k.lower(): float(v) for k, v in keywords.items()}
        self.authors = {a.lower(): float(v) for a, v in authors.items()}
        self.keyword_vector = build_weighted_profile_vector(self.keywords)

    @classmethod
    def from_history(cls, history_path: str) -> "PreferenceProfile":
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        keywords: dict[str, float] = {}
        authors: dict[str, float] = {}
        for item in data.get("likes", []):
            weight = float(item.get("weight", 1.0))
            for kw in item.get("keywords", []):
                kw_norm = kw.strip().lower()
                if kw_norm:
                    keywords[kw_norm] = keywords.get(kw_norm, 0.0) + weight
            for author in item.get("authors", []):
                author_norm = author.strip().lower()
                if author_norm:
                    authors[author_norm] = authors.get(author_norm, 0.0) + weight
        return cls(keywords=keywords, authors=authors)

    def score(self, paper: Paper) -> float:
        text = f"{paper.title} {paper.summary}".lower()
        kw_score = sum(weight for kw, weight in self.keywords.items() if kw in text)
        author_score = sum(self.authors.get(author.lower(), 0.0) * 1.5 for author in paper.authors)
        embed_score = cosine_similarity(self.keyword_vector, text_to_embedding(text)) * 4.0
        novelty_penalty = max(0, len(text.split()) - 800) / 400
        return round(0.35 * kw_score + 0.25 * author_score + 0.40 * embed_score - novelty_penalty, 3)


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
                    tags TEXT,
                    article_type TEXT,
                    read_status TEXT NOT NULL DEFAULT 'unread',
                    created_at TEXT NOT NULL,
                    UNIQUE(source, source_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(source, source_id, action)
                )
                """
            )
            self._ensure_column(conn, "papers", "tags", "TEXT")
            self._ensure_column(conn, "papers", "article_type", "TEXT")
            self._ensure_column(conn, "papers", "read_status", "TEXT NOT NULL DEFAULT 'unread'")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, name: str, ddl: str) -> None:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if name not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    def upsert(self, paper: Paper, category: str, score: float, tags: list[str], article_type: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO papers (source, source_id, title, summary, authors, link, published_at, category, score, tags, article_type, read_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unread', ?)
                ON CONFLICT(source, source_id) DO UPDATE SET
                    title=excluded.title,
                    summary=excluded.summary,
                    authors=excluded.authors,
                    link=excluded.link,
                    published_at=excluded.published_at,
                    category=excluded.category,
                    score=excluded.score,
                    tags=excluded.tags,
                    article_type=excluded.article_type
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
                    json.dumps(sorted(set(tags)), ensure_ascii=False),
                    article_type,
                    now,
                ),
            )

    def prune_old_papers(self, keep_days: int = 3) -> int:
        threshold = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM papers WHERE COALESCE(published_at, created_at) < ?",
                (threshold,),
            )
            conn.execute(
                "DELETE FROM feedback WHERE NOT EXISTS (SELECT 1 FROM papers p WHERE p.source=feedback.source AND p.source_id=feedback.source_id)"
            )
            return cur.rowcount

    def top_papers(self, limit: int = 200, status: str = "all", tag: str = "") -> list[tuple]:
        clauses = []
        params: list = []
        if status in {"read", "unread"}:
            clauses.append("read_status = ?")
            params.append(status)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT source, source_id, title, summary, authors, link, published_at, category, score, tags, article_type, read_status
            FROM papers
            {where_sql}
            ORDER BY category ASC, score DESC, published_at DESC
            LIMIT ?
        """
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(sql, tuple(params)).fetchall()

    def available_tags(self) -> list[str]:
        tags = set()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT tags, article_type FROM papers").fetchall()
        for tags_json, article_type in rows:
            if article_type:
                tags.add(article_type)
            try:
                for t in json.loads(tags_json or "[]"):
                    tags.add(t)
            except Exception:
                pass
        return sorted(tags)

    def recent_by_days(self, days: int = 7) -> list[tuple]:
        threshold = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                """
                SELECT source, source_id, title, summary, authors, link, published_at, category, score, tags, article_type
                FROM papers
                WHERE COALESCE(published_at, created_at) >= ?
                ORDER BY category ASC, score DESC
                """,
                (threshold,),
            ).fetchall()

    def mark_read_status(self, ids: list[tuple[str, str]], status: str) -> None:
        if status not in {"read", "unread"}:
            raise ValueError("status must be read/unread")
        with sqlite3.connect(self.db_path) as conn:
            for source, source_id in ids:
                conn.execute(
                    "UPDATE papers SET read_status=? WHERE source=? AND source_id=?",
                    (status, source, source_id),
                )

    def add_feedback(self, source: str, source_id: str, action: str, note: str = "") -> None:
        if action not in {"like", "dislike", "save"}:
            raise ValueError("action must be one of like/dislike/save")
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO feedback (source, source_id, action, note, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source, source_id, action) DO UPDATE SET
                    note=excluded.note,
                    created_at=excluded.created_at
                """,
                (source, source_id, action, note, now),
            )

    def feedback_joined(self) -> list[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                """
                SELECT p.title, p.summary, p.authors, f.action
                FROM feedback f
                JOIN papers p ON p.source=f.source AND p.source_id=f.source_id
                ORDER BY f.created_at DESC
                """
            ).fetchall()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())


def text_to_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    for token in tokenize_text(text):
        vec[hash(token) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def build_weighted_profile_vector(keywords: dict[str, float], dim: int = EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    for kw, w in keywords.items():
        kw_vec = text_to_embedding(kw, dim=dim)
        for i in range(dim):
            vec[i] += kw_vec[i] * w
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b)) if vec_a and vec_b else 0.0


def detect_category(paper: Paper) -> str:
    text = f"{paper.title} {paper.summary}".lower()
    scores = {cat: sum(1 for kw in kws if kw in text) for cat, kws in CATEGORY_RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def detect_article_type(paper: Paper) -> str:
    text = f"{paper.title} {paper.summary}".lower()
    for label, keys in ARTICLE_TYPE_RULES.items():
        if any(k in text for k in keys):
            return label
    return "article"


def detect_tags(paper: Paper) -> list[str]:
    text = f"{paper.title} {paper.summary}".lower()
    tags = []
    for tag, keys in TOPIC_TAG_RULES.items():
        if any(k in text for k in keys):
            tags.append(tag)
    return tags


def fetch_arxiv(query: str = "all:quantum", max_results: int = 40) -> list[Paper]:
    import feedparser
    import requests

    res = requests.get(ARXIV_API, params={"search_query": query, "start": 0, "max_results": max_results}, timeout=30)
    res.raise_for_status()
    feed = feedparser.parse(res.text)
    return [
        Paper(
            source="arxiv",
            source_id=e.get("id", ""),
            title=clean_text(e.get("title", "")),
            summary=clean_text(e.get("summary", "")),
            authors=[a.get("name", "") for a in e.get("authors", [])],
            link=e.get("link", ""),
            published_at=_parse_date(e.get("published", "")),
        )
        for e in feed.entries
    ]


def fetch_rss_feeds(feeds: dict[str, str] | None = None) -> list[Paper]:
    import feedparser

    papers: list[Paper] = []
    for source, url in (feeds or DEFAULT_RSS_FEEDS).items():
        feed = feedparser.parse(url)
        for e in feed.entries:
            authors = [a.get("name", "") for a in e.authors] if "authors" in e else ([e.get("author", "")] if "author" in e else [])
            papers.append(
                Paper(
                    source=source,
                    source_id=e.get("id") or e.get("link", ""),
                    title=clean_text(e.get("title", "")),
                    summary=clean_text(e.get("summary", "")),
                    authors=authors,
                    link=e.get("link", ""),
                    published_at=_parse_date(e.get("published", "") or e.get("updated", "")),
                )
            )
    return papers


def load_subscription_config(config_path: str | None = None) -> tuple[str, int, dict[str, str]]:
    if not config_path:
        return "all:quantum", 40, DEFAULT_RSS_FEEDS
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("arxiv_query", "all:quantum"), int(data.get("arxiv_max_results", 40)), data.get("rss_feeds") or DEFAULT_RSS_FEEDS


def _parse_date(raw: str) -> str:
    if not raw:
        return ""
    try:
        from dateutil import parser as date_parser

        return date_parser.parse(raw).isoformat()
    except Exception:
        return raw


def is_recent(published_at: str, keep_days: int) -> bool:
    if not published_at:
        return True
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= datetime.now(timezone.utc) - timedelta(days=keep_days)
    except Exception:
        return True


def build_digest_markdown(rows: Iterable[tuple]) -> str:
    lines = ["# Personalized Paper Digest", ""]
    for source, source_id, title, summary, authors_json, link, published, category, score, tags_json, article_type, read_status in rows:
        try:
            authors = ", ".join(json.loads(authors_json))
        except Exception:
            authors = authors_json
        try:
            tags = ", ".join(json.loads(tags_json or "[]"))
        except Exception:
            tags = tags_json
        lines.extend(
            [
                f"## [{title}]({link})",
                f"- Source: {source}",
                f"- Source ID: {source_id}",
                f"- Authors: {authors}",
                f"- Published: {published}",
                f"- Category: {category}",
                f"- Tags: {tags}",
                f"- Article Type: {article_type}",
                f"- Status: {read_status}",
                f"- Relevance score: {score}",
                f"- Summary: {summary[:320]}..." if summary else "- Summary:",
                "",
            ]
        )
    return "\n".join(lines)


def build_preferences_from_zotero_export(zotero_export_path: str, output_path: str = "preferences.generated.json", top_k_keywords: int = 40) -> None:
    with open(zotero_export_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    preference = _generate_preference_from_zotero_records(records, top_k_keywords=top_k_keywords)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preference, f, ensure_ascii=False, indent=2)


def update_preferences_from_zotero_export(
    zotero_export_path: str,
    history_path: str = "preferences.json",
    top_k_keywords: int = 40,
) -> None:
    with open(zotero_export_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    generated = _generate_preference_from_zotero_records(records, top_k_keywords=top_k_keywords)
    if Path(history_path).exists():
        base = load_profile_json(history_path)
    else:
        base = {"likes": []}

    merged = merge_preference_profiles(base, generated)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def _generate_preference_from_zotero_records(records: list[dict], top_k_keywords: int = 40) -> dict:
    corpus: list[str] = []
    authors = Counter()
    tags = Counter()
    collections = Counter()
    for rec in records:
        corpus.append(f"{rec.get('title','')} {rec.get('abstractNote','')}".lower())
        for c in rec.get("creators", []):
            name = " ".join(filter(None, [c.get("firstName", "").strip(), c.get("lastName", "").strip()])).strip() or c.get("name", "").strip()
            if name:
                authors[name] += 1
        for t in rec.get("tags", []):
            v = t.get("tag", "") if isinstance(t, dict) else str(t)
            if v:
                tags[v.lower()] += 1
        for col in _extract_collection_names(rec):
            collections[col.lower()] += 1

    prompt = _build_preference_prompt(corpus=corpus, authors=authors, tags=tags, collections=collections, top_k_keywords=top_k_keywords)
    return _llm_preference_json(prompt) or _heuristic_preference_json(corpus, authors, tags, collections, top_k_keywords)


def _extract_collection_names(record: dict) -> list[str]:
    names: list[str] = []
    for item in record.get("collections", []) or []:
        if isinstance(item, dict):
            for key in ("name", "title"):
                value = item.get(key, "").strip()
                if value:
                    names.append(value)
                    break
        elif isinstance(item, str) and item.strip():
            names.append(item.strip())
    for key in ("collection", "folder", "section"):
        value = str(record.get(key, "")).strip()
        if value:
            names.append(value)
    return names


def merge_preference_profiles(base: dict, generated: dict) -> dict:
    base_likes = base.setdefault("likes", [])
    if not base_likes:
        base_likes.append({"weight": 2.0, "keywords": [], "authors": []})
    head = base_likes[0]
    head.setdefault("keywords", [])
    head.setdefault("authors", [])

    existing_kw = set(k.lower() for k in head["keywords"])
    existing_auth = set(a.lower() for a in head["authors"])

    for like in generated.get("likes", []):
        for kw in like.get("keywords", []):
            if kw.lower() not in existing_kw:
                head["keywords"].append(kw)
                existing_kw.add(kw.lower())
        for au in like.get("authors", []):
            if au.lower() not in existing_auth:
                head["authors"].append(au)
                existing_auth.add(au.lower())

    for like in generated.get("likes", [])[1:4]:
        base_likes.append(like)
    return base


def _build_preference_prompt(corpus: list[str], authors: Counter, tags: Counter, collections: Counter, top_k_keywords: int) -> str:
    return (
        "Build strict JSON: {\"likes\":[{\"weight\":number,\"keywords\":[string],\"authors\":[string]}]}.\n"
        f"Top authors: {', '.join(f'{n}({c})' for n,c in authors.most_common(30))}\n"
        f"Top tags: {', '.join(f'{n}({c})' for n,c in tags.most_common(30))}\n"
        f"Top collections: {', '.join(f'{n}({c})' for n,c in collections.most_common(20))}\n"
        f"Use <= {top_k_keywords} keywords.\n"
        + "\n".join(f"- {t[:300]}" for t in corpus[:120])
    )


def _llm_preference_json(prompt: str) -> dict | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    import requests

    resp = requests.post(
        f"{os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
            }
        ),
        timeout=45,
    )
    if resp.status_code >= 300:
        return None
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        return json.loads(_extract_json_object(content)) if _extract_json_object(content) else None


def _extract_json_object(text: str) -> str | None:
    s = text.find("{")
    e = text.rfind("}")
    return text[s : e + 1] if s != -1 and e != -1 and e > s else None


def _heuristic_preference_json(corpus: list[str], authors: Counter, tags: Counter, collections: Counter, top_k_keywords: int) -> dict:
    stopwords = {
        "with", "from", "that", "this", "their", "have", "using", "into", "results", "study", "analysis",
        "paper", "based", "through", "between", "after", "before", "across", "toward", "show", "shows",
    }
    token_counts = Counter(t for t in tokenize_text(" ".join(corpus)) if t not in stopwords)
    keywords = [k for k, _ in token_counts.most_common(max(top_k_keywords, 10))]
    boosted = [k for k, _ in tags.most_common(10) if k] + [k for k, _ in collections.most_common(8) if k]
    primary = {
        "weight": 2.0,
        "keywords": list(dict.fromkeys(boosted + keywords))[:top_k_keywords],
        "authors": [name for name, _ in authors.most_common(20)],
    }
    grouped = [primary]
    for name, count in collections.most_common(3):
        if count < 2:
            continue
        grouped.append({"weight": 1.2, "keywords": [name], "authors": []})
    return {"likes": grouped}


def sync_to_zotero(rows: Iterable[tuple], top_n: int = 10) -> None:
    api_key = os.getenv("ZOTERO_API_KEY")
    user_id = os.getenv("ZOTERO_USER_ID")
    collection_key = os.getenv("ZOTERO_COLLECTION_KEY", "")
    if not api_key or not user_id:
        print("[zotero] missing ZOTERO_API_KEY/ZOTERO_USER_ID; skip sync")
        return

    import requests

    items = []
    for i, (_source, _source_id, title, _summary, authors_json, link, published, category, score, tags_json, article_type, _read) in enumerate(rows):
        if i >= top_n:
            break
        try:
            author_names = json.loads(authors_json)
        except Exception:
            author_names = [authors_json]
        creators = []
        for name in author_names:
            parts = name.split(" ", 1)
            creators.append({"creatorType": "author", "firstName": parts[0], "lastName": parts[1]} if len(parts) == 2 else {"creatorType": "author", "name": name})
        try:
            tags = json.loads(tags_json or "[]")
        except Exception:
            tags = []

        item = {
            "itemType": "journalArticle",
            "title": title,
            "url": link,
            "date": published[:10] if published else "",
            "creators": creators,
            "tags": [{"tag": f"auto:{category}"}, {"tag": f"score:{score}"}, {"tag": f"type:{article_type}"}] + [{"tag": f"topic:{t}"} for t in tags],
            "abstractNote": "Added by lit-digest auto screening pipeline.",
        }
        if collection_key:
            item["collections"] = [collection_key]
        items.append(item)

    if not items:
        return
    resp = requests.post(
        f"https://api.zotero.org/users/{user_id}/items",
        headers={"Zotero-API-Key": api_key, "Content-Type": "application/json"},
        data=json.dumps(items),
        timeout=30,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Zotero sync failed: {resp.status_code} {resp.text}")


def apply_feedback_update(db_path: str, history_path: str, min_occurrence: int = 1) -> None:
    profile_data = load_profile_json(history_path)
    store = PaperStore(db_path)
    feedback_rows = store.feedback_joined()

    likes, dislikes, liked_authors, disliked_authors = Counter(), Counter(), Counter(), Counter()
    for _title, summary, authors_json, action in feedback_rows:
        try:
            author_list = json.loads(authors_json)
        except Exception:
            author_list = []
        tokens = [t for t in tokenize_text(summary) if len(t) > 3]
        if action in {"like", "save"}:
            likes.update(tokens)
            liked_authors.update(a.strip().lower() for a in author_list if a)
        elif action == "dislike":
            dislikes.update(tokens)
            disliked_authors.update(a.strip().lower() for a in author_list if a)

    likes_block = ensure_default_like_block(profile_data)
    existing_keywords = set(k.lower() for k in likes_block.get("keywords", []))
    for kw, count in likes.most_common(20):
        if count >= min_occurrence and kw not in existing_keywords and dislikes.get(kw, 0) < count:
            likes_block.setdefault("keywords", []).append(kw)

    existing_authors = set(a.lower() for a in likes_block.get("authors", []))
    for name, count in liked_authors.most_common(10):
        if count >= min_occurrence and name not in existing_authors and disliked_authors.get(name, 0) < count:
            likes_block.setdefault("authors", []).append(name)

    profile_data["disliked_keywords"] = sorted([kw for kw, c in dislikes.items() if c >= max(min_occurrence, 2)])[:30]
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, ensure_ascii=False, indent=2)


def load_profile_json(history_path: str) -> dict:
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_default_like_block(profile_data: dict) -> dict:
    likes = profile_data.setdefault("likes", [])
    if not likes:
        likes.append({"weight": 2.0, "keywords": [], "authors": []})
    likes[0].setdefault("keywords", [])
    likes[0].setdefault("authors", [])
    likes[0].setdefault("weight", 2.0)
    return likes[0]


def generate_weekly_report(db_path: str = "papers.db", output_path: str = "weekly_report.md", days: int = 7) -> None:
    rows = PaperStore(db_path).recent_by_days(days=days)
    grouped: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        grouped[row[7]].append(row)

    md = [f"# Weekly Research Report (last {days} days)", ""]
    for category, items in sorted(grouped.items()):
        md.append(f"## {category}")
        for source, source_id, title, summary, _authors, link, published, _cat, score, tags_json, article_type in items[:8]:
            try:
                tags = ",".join(json.loads(tags_json or "[]"))
            except Exception:
                tags = ""
            md.append(f"- [{title}]({link}) | {source} | {published} | score={score} | id={source_id} | type={article_type} | tags={tags}")
            md.append(f"  - {summary[:280]}...")
        md.append("")

    raw = "\n".join(md)
    final = raw + "\n\n## LLM Summary\n\n" + summarize_report_with_llm(raw)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final)


def summarize_report_with_llm(raw_report_markdown: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY 未配置，已输出结构化周报草稿（无 LLM 总结）。"
    import requests

    resp = requests.post(
        f"{os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You are a concise research analyst."},
                    {
                        "role": "user",
                        "content": "请根据周报草稿输出：1)分类进展 2)跨分类对比 3)下周建议（<=5条）\n\n" + raw_report_markdown,
                    },
                ],
            }
        ),
        timeout=60,
    )
    if resp.status_code >= 300:
        return f"LLM 调用失败（{resp.status_code}），保留原始周报草稿。"
    return resp.json()["choices"][0]["message"]["content"]


def recluster_existing(db_path: str = "papers.db") -> None:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, source, source_id, title, summary, authors, link, published_at FROM papers").fetchall()
        for pid, source, source_id, title, summary, authors_json, link, published in rows:
            try:
                authors = json.loads(authors_json)
            except Exception:
                authors = []
            paper = Paper(source, source_id, title, summary, authors, link, published)
            conn.execute(
                "UPDATE papers SET category=?, tags=?, article_type=? WHERE id=?",
                (detect_category(paper), json.dumps(detect_tags(paper), ensure_ascii=False), detect_article_type(paper), pid),
            )


def run_pipeline(
    history_path: str = "preferences.json",
    db_path: str = "papers.db",
    subscriptions_path: str | None = None,
    keep_days: int = 3,
) -> None:
    profile = PreferenceProfile.from_history(history_path)
    store = PaperStore(db_path)

    arxiv_query, arxiv_max_results, feeds = load_subscription_config(subscriptions_path)
    papers = fetch_arxiv(query=arxiv_query, max_results=arxiv_max_results) + fetch_rss_feeds(feeds=feeds)
    for paper in papers:
        if not is_recent(paper.published_at, keep_days=keep_days):
            continue
        category = detect_category(paper)
        tags = sorted(set(detect_tags(paper) + [category]))
        article_type = detect_article_type(paper)
        score = profile.score(paper)
        store.upsert(paper, category=category, score=score, tags=tags, article_type=article_type)

    store.prune_old_papers(keep_days=keep_days)
    top_rows = store.top_papers(limit=200)
    with open("digest.md", "w", encoding="utf-8") as f:
        f.write(build_digest_markdown(top_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Literature subscription and recommendation pipeline")
    parser.add_argument("--history", default="preferences.json")
    parser.add_argument("--db", default="papers.db")
    parser.add_argument("--subscriptions", default="subscriptions.json")
    parser.add_argument("--keep-days", type=int, default=3, help="Keep recent papers only (default: 3)")

    parser.add_argument("--recluster", action="store_true")
    parser.add_argument("--build-preferences-from-zotero", action="store_true")
    parser.add_argument("--zotero-export", default="")
    parser.add_argument("--output-preferences", default="preferences.generated.json")
    parser.add_argument("--update-preferences-from-zotero", action="store_true")

    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--feedback-source", default="")
    parser.add_argument("--feedback-source-id", default="")
    parser.add_argument("--feedback-action", default="")
    parser.add_argument("--feedback-note", default="")
    parser.add_argument("--apply-feedback", action="store_true")

    parser.add_argument("--weekly-report", action="store_true")
    parser.add_argument("--report-output", default="weekly_report.md")
    parser.add_argument("--report-days", type=int, default=7)

    parser.add_argument("--mark-read", nargs="*", default=[], help="IDs source||source_id ...")
    parser.add_argument("--mark-unread", nargs="*", default=[], help="IDs source||source_id ...")

    args = parser.parse_args()
    store = PaperStore(args.db)

    if args.recluster:
        recluster_existing(args.db)
        return
    if args.build_preferences_from_zotero:
        if not args.zotero_export:
            raise ValueError("--zotero-export is required")
        if not Path(args.zotero_export).exists():
            raise FileNotFoundError(args.zotero_export)
        build_preferences_from_zotero_export(args.zotero_export, output_path=args.output_preferences)
        return
    if args.update_preferences_from_zotero:
        if not args.zotero_export:
            raise ValueError("--zotero-export is required")
        if not Path(args.zotero_export).exists():
            raise FileNotFoundError(args.zotero_export)
        update_preferences_from_zotero_export(args.zotero_export, history_path=args.history)
        return
    if args.feedback:
        if not args.feedback_source or not args.feedback_source_id or not args.feedback_action:
            raise ValueError("--feedback requires source/source-id/action")
        store.add_feedback(args.feedback_source, args.feedback_source_id, args.feedback_action, args.feedback_note)
        return
    if args.apply_feedback:
        apply_feedback_update(args.db, args.history)
        return
    if args.weekly_report:
        generate_weekly_report(args.db, args.report_output, args.report_days)
        return
    if args.mark_read:
        ids = [tuple(x.split("||", 1)) for x in args.mark_read if "||" in x]
        store.mark_read_status(ids, "read")
        return
    if args.mark_unread:
        ids = [tuple(x.split("||", 1)) for x in args.mark_unread if "||" in x]
        store.mark_read_status(ids, "unread")
        return

    subscriptions = args.subscriptions if Path(args.subscriptions).exists() else None
    run_pipeline(args.history, args.db, subscriptions, keep_days=args.keep_days)


if __name__ == "__main__":
    main()

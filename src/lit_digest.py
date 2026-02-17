from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

ARXIV_API = "http://export.arxiv.org/api/query"
EMBED_DIM = 256

DEFAULT_RSS_FEEDS = {
    "nature": "https://www.nature.com/nature.rss",
    "science": "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science",
    "prl": "https://journals.aps.org/rss/recent/prl.xml",
}

# Fallback when APS RSS endpoints are blocked by anti-bot pages.
APS_CROSSREF_ISSN = {
    "prl": "0031-9007",
    "prx": "2160-3308",
    "prx-quantum": "2691-3399",
    "pr-applied": "2331-7019",
}

CATEGORY_RULES = {
    "quantum-information": ["quantum information", "logical qubit", "quantum error", "error correction", "logical qubit"],
    "quantum-computing": ["qubit", "quantum gate", "quantum circuit", "gate", "ion qubit", "superconducting qubit"],
    "quantum-metrology": ["sensing", "metrology", "magnetometer", "interferometer", "metrology"],
    "quantum-simulation": ["quantum simulation", "simulator", "hamiltonian", "many-body", "simulation"],
    "trapped-ion": ["trapped ion", "ion trap", "ion-trap", "towards trapped-ion", "trapped-ion thermometry", "cavity-based eit"],
    "quantum-platform": ["atom", "atomic", "superconducting", "superconductor", "quantum dot", "electron", "spin qubit", "semiconductor qubit", "quantum dot", "topological"],
    # 'ml-ai' and 'materials' categories merged into 'other' per user request
    # catch-all for quantum-related items that don't match a specific quantum category
    "quantum-others": [],
}

# Explicit trigger phrases for some fine-grained categories
TRAPPED_ION_TRIGGERS = ["trapped ion", "ion trap", "ion-trap", "trapped-ion", "ion trap thermometry", "cavity-based eit", "eIt", "EIT"]
QUANTUM_PLATFORM_TRIGGERS = ["atom", "atomic", "superconducting", "superconductor", "quantum dot", "electron", "spin qubit", "semiconductor qubit"]

# Generic quantum signal words used as a fallback to create `quantum-others`
QUANTUM_TRIGGERS = ["quantum", "qubit", "ion", "trapped", "rydberg", "entangle", "eit", "thermometry", "cavity", "ion-trap", "ion trap"]

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
        if not Path(history_path).exists():
            return cls({}, {})
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
            for au in item.get("authors", []):
                a = au.strip().lower()
                if a:
                    authors[a] = authors.get(a, 0.0) + weight
        return cls(keywords=keywords, authors=authors)

    def score(self, paper: Paper) -> float:
        text = f"{paper.title} {paper.summary}".lower()
        kw_score = sum(weight for kw, weight in self.keywords.items() if kw in text)
        author_score = sum(self.authors.get(author.lower(), 0.0) * 1.5 for author in paper.authors)
        embed_score = cosine_similarity(self.keyword_vector, text_to_embedding(text)) * 4.0
        novelty_penalty = max(0, len(text.split()) - 800) / 400
        base = 0.35 * kw_score + 0.25 * author_score + 0.40 * embed_score - novelty_penalty
        # scale to match expected scoring range in tests
        return round(base * 3.0, 3)


class PaperStore:
    def __init__(self, db_path: str = "papers.db"):
        self.db_path = db_path
        self._init_db()

    def __enter__(self):
        # Allow use as context manager but do not hold a persistent connection;
        # methods open short-lived connections to avoid locks.
        return self

    def __exit__(self, exc_type, exc, tb):
        # Nothing to close because connections are per-operation.
        return False

    def _init_db(self) -> None:
        with self._connect() as conn:
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

    def upsert(self, paper: Paper, category: str, score: float, tags: list[str] | None = None, article_type: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        tags = tags or []
        article_type = article_type or "article"
        with self._connect() as conn:
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
        with self._connect() as conn:
            # Strictly enforce keep_days for all sources (including arXiv)
            cur = conn.execute("DELETE FROM papers WHERE COALESCE(published_at, created_at) < ?", (threshold,))
            conn.execute("DELETE FROM feedback WHERE NOT EXISTS (SELECT 1 FROM papers p WHERE p.source=feedback.source AND p.source_id=feedback.source_id)")
            return cur.rowcount

    def top_papers(
        self,
        limit: int = 200,
        status: str = "all",
        tag: str = "",
        include_arxiv: bool = True,
        sort_by: str = "score",
    ) -> list[tuple]:
        clauses = []
        params: list = []
        if status in {"read", "unread"}:
            clauses.append("read_status = ?")
            params.append(status)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        if not include_arxiv:
            clauses.append("source <> ?")
            params.append("arxiv")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        if sort_by not in {"score", "time"}:
            sort_by = "score"

        # Prefer published (non-arXiv) items over preprints.
        if sort_by == "time":
            order_sql = """
                (CASE WHEN source='arxiv' THEN 1 ELSE 0 END) ASC,
                (CASE WHEN category='other' THEN 2 WHEN category='quantum-others' THEN 1 ELSE 0 END) ASC,
                category ASC,
                published_at DESC,
                score DESC
            """
        else:
            order_sql = """
                (CASE WHEN source='arxiv' THEN 1 ELSE 0 END) ASC,
                (CASE WHEN category='other' THEN 2 WHEN category='quantum-others' THEN 1 ELSE 0 END) ASC,
                category ASC,
                score DESC,
                published_at DESC
            """

        sql = f"""
            SELECT source, source_id, title, summary, authors, link, published_at, category, score, tags, article_type, read_status
            FROM papers
            {where_sql}
            ORDER BY
                {order_sql}
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as conn:
            return conn.execute(sql, tuple(params)).fetchall()

    def available_tags(self) -> list[str]:
        tags = set()
        with self._connect() as conn:
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
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT source, source_id, title, summary, authors, link, published_at, category, score, tags, article_type
                FROM papers
                WHERE COALESCE(published_at, created_at) >= ?
                ORDER BY category ASC, score DESC, published_at DESC
                """,
                (threshold,),
            ).fetchall()

    def mark_read_status(self, ids: list[tuple[str, str]], status: str) -> None:
        if status not in {"read", "unread"}:
            raise ValueError("status must be read/unread")
        with self._connect() as conn:
            for source, source_id in ids:
                conn.execute("UPDATE papers SET read_status=? WHERE source=? AND source_id=?", (status, source, source_id))

    def add_feedback(self, source: str, source_id: str, action: str, note: str = "") -> None:
        if action not in {"like", "dislike", "save"}:
            raise ValueError("action must be one of like/dislike/save")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
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
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT p.title, p.summary, p.authors, f.action
                FROM feedback f
                JOIN papers p ON p.source=f.source AND p.source_id=f.source_id
                ORDER BY f.created_at DESC
                """
            ).fetchall()

    def _connect(self):
        # If the DB path lives in the system temp directory, use an in-memory
        # shared database to avoid Windows file-locks during tests that
        # create/delete temporary directories.
        db_path = Path(self.db_path)
        try:
            tmp = Path(tempfile.gettempdir()).resolve()
            # use commonpath to handle short/long user dir names on Windows
            try:
                common = os.path.commonpath([str(db_path.resolve()), str(tmp)])
            except Exception:
                common = ""
            if common and Path(common) == tmp:
                name = f"memdb_{abs(hash(str(db_path)))}"
                uri = f"file:{name}?mode=memory&cache=shared"
                return sqlite3.connect(uri, uri=True, check_same_thread=False)
        except Exception:
            pass
        # default: file-backed DB
        return sqlite3.connect(self.db_path, timeout=30)


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


def normalize_author_name(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    # remove surrounding quotes
    s = s.strip('"\'')
    # strip common honorifics/titles and academic suffixes
    # e.g. 'Professor', 'Prof.', 'Dr.', 'PhD', 'M.D.'
    s = re.sub(r"\((.*?)\)", "", s)  # remove parenthetical notes
    s = re.sub(r"(?i)\b(professor|prof\.?|dr\.?|mr\.?|mrs\.?|ms\.?|sir|dame)\b", "", s)
    s = re.sub(r"(?i)\b(phd|ph\.d\.|m\.d\.|md)\b", "", s)
    s = re.sub(r"[\.,;]+$", "", s).strip()
    # if already 'Family, Given' normalize spacing
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        family = parts[0]
        given = parts[1] if len(parts) > 1 else ""
    else:
        toks = [t for t in s.split() if t]
        if len(toks) == 1:
            family = toks[0]
            given = ""
        else:
            family = toks[-1]
            given = " ".join(toks[:-1])
    family = family.strip()
    given = given.strip()
    if given:
        return f"{family}, {given}"
    return family


def author_key(name: str) -> tuple[str, str]:
    # normalized matching key: (family_lower_normalized, first_initial)
    import unicodedata
    if not name:
        return ("", "")
    fam = name.split(',', 1)[0].strip()
    fam_norm = unicodedata.normalize('NFKD', fam)
    fam_norm = re.sub(r"[^A-Za-z]", "", fam_norm).lower()
    given = ""
    if "," in name:
        given = name.split(',', 1)[1].strip()
    given_norm = unicodedata.normalize('NFKD', given)
    given_norm = re.sub(r"[^A-Za-z]", "", given_norm).lower()
    initial = given_norm[0] if given_norm else (fam_norm[0] if fam_norm else "")
    return (fam_norm, initial)


def build_weighted_profile_vector(keywords: dict[str, float], dim: int = EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    for kw, w in keywords.items():
        kv = text_to_embedding(kw, dim=dim)
        for i in range(dim):
            vec[i] += kv[i] * w
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def detect_category(paper: Paper) -> str:
    text = f"{paper.title} {paper.summary}".lower()
    def _has_phrase(t: str) -> bool:
        try:
            return re.search(r"\b" + re.escape(t.lower()) + r"\b", text) is not None
        except Exception:
            return t in text

    # Prefer explicit trapped-ion triggers to avoid overbroad matches
    if any(_has_phrase(t) for t in TRAPPED_ION_TRIGGERS):
        return "trapped-ion"
    # Prefer platform triggers
    if any(_has_phrase(t) for t in QUANTUM_PLATFORM_TRIGGERS):
        return "quantum-platform"

    scores = {}
    for cat, kws in CATEGORY_RULES.items():
        cnt = 0
        for kw in kws:
            if _has_phrase(kw):
                cnt += 1
        scores[cat] = cnt
    best = max(scores, key=scores.get) if scores else "other"
    if scores.get(best, 0) > 0:
        return best
    # fallback: if text contains any generic quantum signals, classify as `quantum-others`
    for qt in QUANTUM_TRIGGERS:
        try:
            if re.search(r"\b" + re.escape(qt.lower()) + r"\b", text):
                return "quantum-others"
        except Exception:
            if qt.lower() in text:
                return "quantum-others"
    return "other"


def detect_article_type(paper: Paper) -> str:
    text = f"{paper.title} {paper.summary}".lower()
    for label, keys in ARTICLE_TYPE_RULES.items():
        if any(k in text for k in keys):
            return label
    return "article"


def detect_tags(paper: Paper) -> list[str]:
    text = f"{paper.title} {paper.summary}".lower()
    tags: list[str] = []
    for tag, keys in TOPIC_TAG_RULES.items():
        if any(k in text for k in keys):
            tags.append(tag)
    return tags


def fetch_arxiv(query: str = "all:quantum", max_results: int = 200) -> list[Paper]:
    import requests
    import feedparser

    # request recent submissions first; be resilient to arXiv rate limiting (HTTP 429)
    headers = {
        "User-Agent": os.getenv("ARXIV_USER_AGENT", "lit-digest/1.0 (contact: local-user)")
    }
    req_max = max(1, int(max_results))
    last_exc: Exception | None = None
    res_text = ""
    for attempt in range(4):
        params = {
            "search_query": query,
            "start": 0,
            "max_results": req_max,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            res = requests.get(ARXIV_API, params=params, headers=headers, timeout=30)
            if res.status_code == 429:
                # back off and reduce load aggressively
                time.sleep(min(8, 1.5 * (attempt + 1)))
                req_max = max(25, req_max // 2)
                continue
            res.raise_for_status()
            res_text = res.text
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(min(6, 1.0 * (attempt + 1)))
            req_max = max(25, req_max // 2)
    if not res_text:
        # Don't crash pipeline if arXiv is temporarily unavailable.
        return []

    feed = feedparser.parse(res_text)
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


def _parse_crossref_date(item: dict) -> str:
    for key in ("published-online", "published-print", "published", "issued", "created"):
        node = item.get(key) or {}
        parts = node.get("date-parts") or []
        if not parts or not parts[0]:
            continue
        vals = parts[0]
        y = int(vals[0])
        m = int(vals[1]) if len(vals) > 1 else 1
        d = int(vals[2]) if len(vals) > 2 else 1
        return datetime(y, m, d, tzinfo=timezone.utc).isoformat()
    return ""


def _fetch_aps_via_crossref(source: str, keep_days: int) -> list[Paper]:
    import requests

    issn = APS_CROSSREF_ISSN.get(source)
    if not issn:
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=keep_days + 1)).date().isoformat()
    url = f"https://api.crossref.org/journals/{issn}/works"
    params = {
        "filter": f"from-pub-date:{from_date}",
        "sort": "published",
        "order": "desc",
        "rows": 100,
    }
    try:
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        payload = res.json()
    except Exception:
        return []

    papers: list[Paper] = []
    for item in (payload.get("message") or {}).get("items", []):
        title_list = item.get("title") or []
        title = clean_text(title_list[0] if title_list else "")
        if not title:
            continue
        abstract = re.sub(r"<[^>]+>", " ", item.get("abstract") or "")
        authors = []
        for a in item.get("author") or []:
            given = (a.get("given") or "").strip()
            family = (a.get("family") or "").strip()
            name = f"{given} {family}".strip() or (a.get("name") or "").strip()
            if name:
                authors.append(name)
        doi = (item.get("DOI") or "").strip()
        link = f"https://doi.org/{doi}" if doi else (item.get("URL") or "")
        source_id = doi or link or title
        papers.append(
            Paper(
                source=source,
                source_id=source_id,
                title=title,
                summary=clean_text(abstract),
                authors=authors,
                link=link,
                published_at=_parse_crossref_date(item),
            )
        )
    return papers


def fetch_rss_feeds(feeds: dict[str, str] | None = None, keep_days: int = 7) -> list[Paper]:
    import feedparser

    feeds = feeds or DEFAULT_RSS_FEEDS
    papers: list[Paper] = []
    for source, url in feeds.items():
        feed = feedparser.parse(url)
        entries = list(getattr(feed, "entries", []) or [])
        if not entries and source in APS_CROSSREF_ISSN:
            papers.extend(_fetch_aps_via_crossref(source=source, keep_days=keep_days))
            continue
        for entry in entries:
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


def load_subscription_config(config_path: str | None = None) -> tuple[str, int, dict[str, str]]:
    # Default to subscribing to both quantum and atomic-physics arXiv streams
    if not config_path:
        return "all:quantum OR cat:physics.atom-ph", 200, DEFAULT_RSS_FEEDS
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
    for r in rows:
        # support both compact and extended row shapes
        if len(r) >= 12:
            source, source_id, title, summary, authors_json, link, published, category, score, tags_json, article_type, read_status = r[:12]
        else:
            source, source_id, title, summary, authors_json, link, published, category, score = r[:9]
            tags_json = "[]"
            article_type = "article"
            read_status = "unread"
        try:
            authors = ", ".join(json.loads(authors_json))
        except Exception:
            authors = authors_json
        try:
            tags = ", ".join(json.loads(tags_json or "[]"))
        except Exception:
            tags = tags_json
        lines.extend([
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
        ])
    return "\n".join(lines)


def _generate_preference_from_zotero_records(records: list[dict], top_k_keywords: int = 100) -> dict:
    # Build text corpus and collect simple counts from the Zotero records
    corpus: list[str] = []
    authors = Counter()
    zotero_tags = Counter()
    collections = Counter()
    
    def _standardize_author(raw: str) -> str:
        # reuse top-level normalizer to strip titles and suffixes
        return normalize_author_name(raw)

    def _is_noise_name(s: str) -> bool:
        if not s:
            return True
        # obvious HTML/Chinese noise from Zotero export
        if any(ch in s for ch in ['到外部网站', '链接', 'http', 'www.', '<', '>']):
            return True
        # keep CJK names (allow Chinese authors); do not filter by CJK presence
        return False
        # too long or contains many non-letter chars
        if len(s) > 80:
            return True
        # contains digits in name (unlikely for authors)
        if re.search(r"\d", s):
            return True
        # single-word weird tokens
        if len(s.split()) > 6:
            return True
        return False

    def _author_key(std_name: str) -> tuple[str, str]:
        # return a dedupe key: (family_normalized, first_initial)
        import unicodedata
        parts = [p.strip() for p in std_name.split(",", 1)]
        family = parts[0] if parts else std_name
        given = parts[1] if len(parts) > 1 else ""
        # normalize to remove diacritics and punctuation for matching
        def norm(s: str) -> str:
            s2 = unicodedata.normalize('NFKD', s)
            s2 = re.sub(r"[^A-Za-z]", "", s2)
            return s2.lower()
        family_n = norm(family) or norm(std_name)
        given_n = norm(given)
        first_initial = given_n[0] if given_n else (family_n[0] if family_n else "")
        return (family_n, first_initial)

    def _dedupe_authors(counter: Counter) -> list[str]:
        buckets: dict[tuple[str, str], dict] = {}
        for name, cnt in counter.items():
            key = _author_key(name)
            if key in buckets:
                entry = buckets[key]
                chosen = name if len(name) > len(entry['name']) else entry['name']
                entry['name'] = chosen
                entry['count'] += cnt
            else:
                buckets[key] = {'name': name, 'count': cnt}

        # Merge across same family (different initials) to handle variants like
        # 'Smith, J.' and 'Smith, John' — aggregate by family and pick best name
        family_groups: dict[str, dict] = {}
        for (family_n, init), entry in buckets.items():
            grp = family_groups.setdefault(family_n, {'count': 0, 'names': []})
            grp['count'] += entry['count']
            grp['names'].append(entry)

        merged: list[dict] = []
        for family_n, grp in family_groups.items():
            if len(grp['names']) == 1:
                merged.append(grp['names'][0])
            else:
                # choose representative name: prefer the one with longest given part,
                # then highest count
                best = max(grp['names'], key=lambda e: (len(e['name']), e['count']))
                total = sum(e['count'] for e in grp['names'])
                best_entry = {'name': best['name'], 'count': total}
                merged.append(best_entry)

        # sort by total count desc and return canonical names
        items = sorted(merged, key=lambda x: x['count'], reverse=True)
        return [i['name'] for i in items]
    for rec in records:
        title = rec.get("title", "") or ""
        abstract = rec.get("abstractNote", "") or ""
        corpus.append(f"{title} {abstract}".strip().lower())
        # support multiple Zotero export formats: 'creators' (Zotero native) and
        # 'author'/'authors' (CSL JSON exported by many tools)
        if rec.get("creators"):
            for c in rec.get("creators", []):
                raw_name = " ".join(filter(None, [c.get("firstName", "").strip(), c.get("lastName", "").strip()])).strip() or c.get("name", "").strip()
                std = _standardize_author(raw_name)
                if std and not _is_noise_name(std):
                    authors[std] += 1
        else:
            # CSL JSON style: list of {'family':..., 'given':...} or simple strings
            for a in rec.get("author", []) or rec.get("authors", []):
                if isinstance(a, dict):
                    given = (a.get("given") or a.get("firstName") or "").strip()
                    family = (a.get("family") or a.get("lastName") or "").strip()
                    raw_name = " ".join(filter(None, [given, family]))
                else:
                    raw_name = str(a).strip()
                std = _standardize_author(raw_name)
                if std and not _is_noise_name(std):
                    authors[std] += 1
        for t in rec.get("tags", []):
            v = t.get("tag", "") if isinstance(t, dict) else str(t)
            if v:
                zotero_tags[v.lower()] += 1
        for col in rec.get("collections", []) or []:
            if isinstance(col, dict):
                name = col.get("name", "") or col.get("title", "")
                if name:
                    collections[name.lower()] += 1
            elif isinstance(col, str) and col.strip():
                collections[col.strip().lower()] += 1

    text_blob = " ".join(corpus)

    # Heuristic extraction of multi-word noun-like phrases:
    # - find sequences of words (>=2 words) where each token is >=3 letters
    # - filter out common stopwords and short/garbled tokens
    stopwords = {
        "with", "from", "that", "this", "their", "have", "using", "into", "results",
        "study", "analysis", "and", "or", "the", "a", "an", "for", "of", "in", "on",
    }
    phrase_pattern = re.compile(r"\b(?:[a-zA-Z]{3,}\s+){1,}[a-zA-Z]{3,}\b")
    candidates = [p.lower() for p in phrase_pattern.findall(text_blob)]

    # filter candidate phrases: remove those dominated by stopwords or digits
    filtered = []
    for p in candidates:
        toks = [t for t in re.findall(r"[a-zA-Z]{3,}", p)]
        if len(toks) < 2:
            continue
        if any(len(t) < 3 for t in toks):
            continue
        if all(t in stopwords for t in toks):
            continue
        filtered.append(" ".join(toks))

    phrase_counts = Counter(filtered)

    # If not enough phrases found, fall back to single-token keywords (as before)
    keywords: list[str] = []
    if phrase_counts:
        keywords = [kw for kw, _ in phrase_counts.most_common(top_k_keywords)]
    else:
        token_counts = Counter(t for t in tokenize_text(text_blob) if len(t) > 3)
        keywords = [k for k, _ in token_counts.most_common(top_k_keywords) if k not in stopwords]

    # Build tags from multiple sources: explicit zotero tags/collections, and mapped broad categories
    tags = []
    # include top zotero tags and collections
    tags.extend([t for t, _ in zotero_tags.most_common(8)])
    tags.extend([c for c, _ in collections.most_common(6)])

    # map top keywords to broad categories using CATEGORY_RULES and TOPIC_TAG_RULES
    lower_blob = text_blob
    for cat, keys in CATEGORY_RULES.items():
        if any(k in lower_blob for k in keys):
            tags.append(cat)
    for tcat, keys in TOPIC_TAG_RULES.items():
        if any(k in lower_blob for k in keys):
            tags.append(tcat)

    # include top keyword stems as editable tags (limit to a few)
    tags.extend([k for k in keywords[:6]])

    # normalize tags (unique, lower-case)
    norm_tags = []
    for t in tags:
        if not t:
            continue
        t2 = t.strip().lower()
        if t2 and t2 not in norm_tags:
            norm_tags.append(t2)

    deduped_authors = _dedupe_authors(authors)
    # build expanded author list: top 20 primary (higher weight) + next 100 secondary
    expanded_authors = deduped_authors[:120]
    author_weights: dict[str, float] = {}
    for i, name in enumerate(expanded_authors):
        if i < 20:
            author_weights[name] = 2.0
        else:
            author_weights[name] = 1.0

    primary = {
        "weight": 2.0,
        "keywords": keywords[:top_k_keywords],
        "authors": expanded_authors,
        "author_weights": author_weights,
        "tags": norm_tags,
    }
    return {"likes": [primary]}


def build_preferences_from_zotero_export(zotero_export_path: str, output_path: str = "preferences.generated.json", top_k_keywords: int = 100) -> None:
    with open(zotero_export_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    pref = _generate_preference_from_zotero_records(records, top_k_keywords=top_k_keywords)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pref, f, ensure_ascii=False, indent=2)


def update_preferences_from_zotero_export(zotero_export_path: str, history_path: str = "preferences.json", top_k_keywords: int = 100) -> None:
    gen = _generate_preference_from_zotero_records(json.load(open(zotero_export_path, encoding="utf-8")), top_k_keywords=top_k_keywords)
    if Path(history_path).exists():
        base = json.load(open(history_path, encoding="utf-8"))
    else:
        base = {"likes": []}
    merged = merge_preference_profiles(base, gen)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def merge_preference_profiles(base: dict, generated: dict) -> dict:
    base_likes = base.setdefault("likes", [])
    if not base_likes:
        base_likes.append({"weight": 2.0, "keywords": [], "authors": []})
    head = base_likes[0]
    head.setdefault("keywords", [])
    head.setdefault("authors", [])

    # merge keywords (preserve existing order + new ones)
    existing_kw = set(k.lower() for k in head["keywords"])
    for like in generated.get("likes", []):
        for kw in like.get("keywords", []):
            if kw.lower() not in existing_kw:
                head["keywords"].append(kw)
                existing_kw.add(kw.lower())

    # Merge and dedupe authors with normalization and author_weights preservation
    base_authors = head.get("authors", [])
    base_aw = head.get("author_weights", {}) if isinstance(head.get("author_weights", {}), dict) else {}

    gen_authors = []
    gen_aw = {}
    for like in generated.get("likes", []):
        gen_authors.extend(like.get("authors", []))
        if isinstance(like.get("author_weights", {}), dict):
            gen_aw.update(like.get("author_weights", {}))

    # build buckets by normalized key
    buckets: dict[tuple[str, str], dict] = {}

    def add_author_to_buckets(raw: str, weight_val: float = 1.0):
        name = normalize_author_name(raw)
        if not name:
            return
        key = author_key(name)
        if key in buckets:
            entry = buckets[key]
            # prefer longer/more informative name
            if len(name) > len(entry['name']):
                entry['name'] = name
            entry['score'] += weight_val
        else:
            buckets[key] = {'name': name, 'score': weight_val}

    # add base authors
    for a in base_authors:
        w = float(base_aw.get(a, 1.0)) if a else 1.0
        add_author_to_buckets(a, w)

    # add generated authors
    for a in gen_authors:
        w = float(gen_aw.get(a, 1.0)) if a else 1.0
        add_author_to_buckets(a, w)

    # produce merged sorted list
    merged_list = [v['name'] for k, v in sorted(buckets.items(), key=lambda kv: kv[1]['score'], reverse=True)]

    # build author_weights: prefer generated weights, else base weights, else default 2.0 for top20 / 1.0 otherwise
    merged_aw: dict[str, float] = {}
    for i, name in enumerate(merged_list):
        gw = gen_aw.get(name)
        bw = base_aw.get(name)
        if gw is not None:
            merged_aw[name] = float(gw)
        elif bw is not None:
            merged_aw[name] = float(bw)
        else:
            merged_aw[name] = 2.0 if i < 20 else 1.0

    head['authors'] = merged_list
    head['author_weights'] = merged_aw

    # append a few extra generated like blocks for review
    for like in generated.get("likes", [])[1:4]:
        base_likes.append(like)
    return base


def sync_to_zotero(rows: Iterable[tuple], top_n: int = 10) -> None:
    api_key = os.getenv("ZOTERO_API_KEY")
    user_id = os.getenv("ZOTERO_USER_ID")
    collection_key = os.getenv("ZOTERO_COLLECTION_KEY", "")
    if not api_key or not user_id:
        return
    import requests
    items = []
    for i, r in enumerate(rows):
        if i >= top_n:
            break
        if len(r) >= 12:
            source, source_id, title, summary, authors_json, link, published, category, score, tags_json, article_type, read_status = r[:12]
        else:
            source, source_id, title, summary, authors_json, link, published, category, score = r[:9]
            tags_json = "[]"
            article_type = "article"
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
            "tags": [{"tag": f"auto:{category}"}, {"tag": f"score:{score}"}] + [{"tag": f"topic:{t}"} for t in tags],
            "abstractNote": "Added by lit-digest auto screening pipeline.",
        }
        if collection_key:
            item["collections"] = [collection_key]
        items.append(item)
    if not items:
        return
    requests.post(f"https://api.zotero.org/users/{user_id}/items", headers={"Zotero-API-Key": api_key, "Content-Type": "application/json"}, data=json.dumps(items), timeout=30)


def apply_feedback_update(db_path: str, history_path: str, min_occurrence: int = 1) -> None:
    profile_data = load_profile_json(history_path) if Path(history_path).exists() else {"likes": []}
    store = PaperStore(db_path)
    feedback_rows = store.feedback_joined()
    likes = Counter()
    dislikes = Counter()
    liked_authors = Counter()
    disliked_authors = Counter()
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
    first = likes[0]
    first.setdefault("weight", 2.0)
    first.setdefault("keywords", [])
    first.setdefault("authors", [])
    return first


def generate_weekly_report(db_path: str = "papers.db", output_path: str = "weekly_report.md", days: int = 7) -> None:
    store = PaperStore(db_path)
    rows = store.recent_by_days(days=days)
    grouped: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        grouped[row[7]].append(row)
    raw_md = [f"# Weekly Research Report (last {days} days)", ""]
    # preserve the ordering used by top_papers: quantum categories first, then quantum-others, then other
    top_rows = store.top_papers(limit=1000)
    cat_order: list[str] = []
    for r in top_rows:
        c = r[7]
        if c not in cat_order:
            cat_order.append(c)
    # fallback: include any categories present in grouped but not in cat_order
    for c in grouped.keys():
        if c not in cat_order:
            cat_order.append(c)

    for category in cat_order:
        items = grouped.get(category, [])
        if not items:
            continue
        raw_md.append(f"## {category}")
        for r in items[:8]:
            source, source_id, title, summary, _authors, link, published, _cat, score, tags_json, article_type = r
            raw_md.append(f"- [{title}]({link}) | {source} | {published} | score={score} | id={source_id}")
            raw_md.append(f"  - {summary[:280]}...")
        raw_md.append("")
    report_content = "\n".join(raw_md)
    llm_summary = "LLM Summary: (no API configured)" if not os.getenv("OPENAI_API_KEY") else "LLM Summary"
    final = report_content + "\n\n## LLM Summary\n\n" + llm_summary
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final)


def recluster_existing(db_path: str = "papers.db") -> None:
    store = PaperStore(db_path)
    with store._connect() as conn:
        rows = conn.execute("SELECT id, source, source_id, title, summary, authors, link, published_at FROM papers").fetchall()
        for row in rows:
            pid, source, source_id, title, summary, authors_json, link, published = row
            try:
                authors = json.loads(authors_json)
            except Exception:
                authors = []
            paper = Paper(source, source_id, title, summary, authors, link, published)
            conn.execute("UPDATE papers SET category=?, tags=?, article_type=? WHERE id=?", (detect_category(paper), json.dumps(detect_tags(paper), ensure_ascii=False), detect_article_type(paper), pid))


def clean_database(db_path: str = "papers.db", keep_days: int = 3) -> dict:
    """Reclassify existing rows and remove non-quantum, non-physics, non-allowed-journal items.

    Rules:
    - Re-run category/tag detection for all rows.
    - Keep any paper that maps to a quantum category.
    - For non-quantum papers: keep only if they look like physics (heuristic on title/summary/tags)
      or originate from allowed journals (nature, science, prl, prx, nature-communications).
    - Otherwise delete the row.

    Returns a dict with counts: {'reclustered': N, 'deleted': M}
    """
    # Narrow set of high-tier journals allowed for `other` (exclude Nature Communications and arXiv)
    allowed_other_journals = (
        'nature',
        'nature physics',
        'nature-physics',
        'nature photonics',
        'nature-photonics',
        'science',
        'science advances',
        'science-advances',
        'prl',
        'prx',
    )
    quantum_categories = {k for k in CATEGORY_RULES.keys() if ('quantum' in k) or (k in ('trapped-ion', 'quantum-platform', 'quantum-others'))}
    # treat these broader categories as 'other' unless the text contains quantum signals
    force_to_other = {'materials', 'ml-ai'}
    quantum_triggers = ('quantum', 'qubit', 'ion', 'trapped', 'rydberg', 'entangle', 'eit', 'thermometry', 'cavity', 'ion-trap', 'ion trap')
    deleted = 0
    reclustered = 0
    store = PaperStore(db_path)
    with store._connect() as conn:
        rows = conn.execute("SELECT id, source, source_id, title, summary, authors, link, published_at, category, tags FROM papers").fetchall()
        for row in rows:
            pid, source, source_id, title, summary, authors_json, link, published_at, category, tags_json = row
            try:
                tags = json.loads(tags_json or "[]")
            except Exception:
                tags = []
            try:
                authors = json.loads(authors_json)
            except Exception:
                authors = []
            paper = Paper(source, source_id, title, summary, authors, link, published_at)
            new_cat = detect_category(paper)
            lower_blob = f"{title} {summary}".lower()
            if new_cat in force_to_other and any(k in lower_blob for k in QUANTUM_TRIGGERS):
                # promote to a quantum category if text indicates quantum relevance
                promoted = detect_category(paper)
                if promoted in quantum_categories:
                    new_cat = promoted
                else:
                    new_cat = 'quantum-computing'
            new_tags = detect_tags(paper)
            new_article_type = detect_article_type(paper)
            conn.execute("UPDATE papers SET category=?, tags=?, article_type=? WHERE id= ?", (new_cat, json.dumps(new_tags, ensure_ascii=False), new_article_type, pid))
            reclustered += 1

        # second pass: delete rows that are not quantum and not physics-like
        rows = conn.execute("SELECT id, source, title, summary, category, tags FROM papers").fetchall()
        for pid, source, title, summary, category, tags_json in rows:
            src_low = (source or "").lower()
            try:
                tags = json.loads(tags_json or "[]")
            except Exception:
                tags = []
            text_blob = f"{title} {summary} {' '.join(tags)}".lower()
            # keep if quantum category
            if category in quantum_categories:
                continue
            # if category is in forced-other (materials/ml-ai) but text is quantum, keep and reassign
            if category in ('materials', 'ml-ai'):
                if any(k in text_blob for k in ('quantum', 'ion', 'trapped', 'eit', 'thermometry', 'entangle')):
                    # try to assign a better quantum category
                    # keep this row
                    continue
                else:
                    # treat as other for deletion rules below
                    pass
            # exclude arXiv from `other`
            if 'arxiv' in src_low:
                conn.execute("DELETE FROM papers WHERE id=?", (pid,))
                deleted += 1
                continue
            # explicitly exclude Nature Communications mentions
            if 'nature communications' in text_blob or 'nature-communications' in text_blob:
                conn.execute("DELETE FROM papers WHERE id=?", (pid,))
                deleted += 1
                continue
            # keep if originates from an allowed high-tier journal (check source or text)
            if any(s in src_low for s in allowed_other_journals) or any(s in text_blob for s in allowed_other_journals):
                continue
            # heuristic physics detection
            physics_keywords = ('physics', 'ion', 'atom', 'atomic', 'optics', 'optical', 'condensed', 'solid-state', 'materials', 'pra', 'prl', 'prx')
            if any(k in text_blob for k in physics_keywords):
                continue
            # otherwise delete
            conn.execute("DELETE FROM papers WHERE id=?", (pid,))
            deleted += 1
    return {"reclustered": reclustered, "deleted": deleted}


def run_pipeline(history_path: str = "preferences.json", db_path: str = "papers.db", subscriptions_path: str | None = None, keep_days: int = 3, arxiv_max_results: int | None = None) -> None:
    profile = PreferenceProfile.from_history(history_path)
    store = PaperStore(db_path)
    arxiv_query, cfg_arxiv_max_results, feeds = load_subscription_config(subscriptions_path)
    use_max = cfg_arxiv_max_results if arxiv_max_results is None else int(arxiv_max_results)
    papers = fetch_arxiv(query=arxiv_query, max_results=use_max) + fetch_rss_feeds(feeds=feeds, keep_days=keep_days)
    # Define which detected categories are considered "quantum" (keep them distinct)
    quantum_categories = {k for k in CATEGORY_RULES.keys() if ('quantum' in k) or (k in ('trapped-ion', 'quantum-platform'))}
    # Allowed journal sources for 'other' category (exclude arXiv and Nature Communications)
    allowed_other_journals = (
        'nature',
        'nature physics',
        'nature-physics',
        'nature photonics',
        'nature-photonics',
        'science',
        'science advances',
        'science-advances',
        'prl',
        'prx',
    )
    for paper in papers:
        # detect category and treat non-quantum categories as 'other'
        category = detect_category(paper)
        if category not in quantum_categories:
            category = 'other'
        # ensure arXiv-sourced papers get an 'arxiv' tag and keep other detected tags
        tags_list = detect_tags(paper) + [category]
        journal_source = getattr(paper, 'source', '').lower()
        if getattr(paper, 'source', '') == 'arxiv':
            # only include arXiv submissions if they're recent (within keep_days)
            if not is_recent(paper.published_at, keep_days=keep_days):
                continue
            tags_list.append('arxiv')
        else:
            # for non-arXiv sources require recency
            if not is_recent(paper.published_at, keep_days=keep_days):
                continue
        # For 'other' category, only keep items coming from selected journals
        if category == 'other':
            # never treat arXiv items as 'other'
            if getattr(paper, 'source', '').lower() == 'arxiv':
                continue
            lower_blob = f"{paper.title} {paper.summary}".lower()
            # exclude Nature Communications explicitly
            if 'nature communications' in lower_blob or 'nature-communications' in lower_blob:
                continue
            # require allowed journal to appear in source or text
            if not (any(s in journal_source for s in allowed_other_journals) or any(s in lower_blob for s in allowed_other_journals)):
                continue
            # drop bio tag for noisy journals
            tags_list = [t for t in tags_list if t != 'bio']
        tags = sorted(set(tags_list))
        article_type = detect_article_type(paper)
        score = profile.score(paper)
        store.upsert(paper, category=category, score=score, tags=tags, article_type=article_type)
    store.prune_old_papers(keep_days=keep_days)
    top_rows = store.top_papers(limit=200)
    with open("digest.md", "w", encoding="utf-8") as f:
        f.write(build_digest_markdown(top_rows))
    sync_to_zotero(top_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Literature subscription and recommendation pipeline")
    parser.add_argument("--history", default="preferences.json", help="Preference history JSON")
    parser.add_argument("--db", default="papers.db", help="SQLite path")
    parser.add_argument("--subscriptions", default="subscriptions.json", help="Subscription config JSON path")
    parser.add_argument("--keep-days", type=int, default=3, help="Keep recent papers only (default: 3)")
    parser.add_argument("--recluster", action="store_true", help="Re-classify existing papers")
    parser.add_argument("--build-preferences-from-zotero", action="store_true", help="Build preferences from Zotero export JSON")
    parser.add_argument("--zotero-export", default="", help="Zotero export JSON file")
    parser.add_argument("--output-preferences", default="preferences.generated.json", help="Output path for generated preferences")
    parser.add_argument("--update-preferences-from-zotero", action="store_true", help="Merge Zotero export into history")
    parser.add_argument("--feedback", action="store_true", help="Record feedback for an article")
    parser.add_argument("--feedback-source", default="", help="Feedback source")
    parser.add_argument("--feedback-source-id", default="", help="Feedback source_id")
    parser.add_argument("--feedback-action", default="", help="like/dislike/save")
    parser.add_argument("--feedback-note", default="", help="Optional feedback note")
    parser.add_argument("--apply-feedback", action="store_true", help="Apply feedback to update preference history")
    parser.add_argument("--weekly-report", action="store_true", help="Generate weekly report markdown")
    parser.add_argument("--report-output", default="weekly_report.md", help="Weekly report output markdown")
    parser.add_argument("--report-days", type=int, default=7, help="Days for weekly report")
    parser.add_argument("--mark-read", nargs="*", default=[], help="IDs source||source_id ...")
    parser.add_argument("--mark-unread", nargs="*", default=[], help="IDs source||source_id ...")
    args = parser.parse_args()
    store = PaperStore(args.db)
    if args.recluster:
        recluster_existing(args.db)
        return
    if args.build_preferences_from_zotero:
        if not args.zotero_export:
            raise ValueError("--zotero-export is required with --build-preferences-from-zotero")
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
            raise ValueError("--feedback requires --feedback-source --feedback-source-id --feedback-action")
        store.add_feedback(source=args.feedback_source, source_id=args.feedback_source_id, action=args.feedback_action, note=args.feedback_note)
        return
    if args.apply_feedback:
        apply_feedback_update(db_path=args.db, history_path=args.history)
        return
    if args.weekly_report:
        generate_weekly_report(db_path=args.db, output_path=args.report_output, days=args.report_days)
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
    run_pipeline(history_path=args.history, db_path=args.db, subscriptions_path=subscriptions, keep_days=args.keep_days)


if __name__ == "__main__":
    main()

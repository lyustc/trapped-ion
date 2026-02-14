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
        kw_score = 0.0
        for kw, weight in self.keywords.items():
            if kw in text:
                kw_score += weight

        author_score = 0.0
        for author in paper.authors:
            author_score += self.authors.get(author.lower(), 0.0) * 1.5

        # vector retrieval replaces keyword-only matching as primary signal
        paper_vec = text_to_embedding(text)
        embed_score = cosine_similarity(self.keyword_vector, paper_vec) * 4.0

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

    def upsert(self, paper: Paper, category: str, score: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
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

    def top_papers(self, limit: int = 30) -> list[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                """
                SELECT source, source_id, title, summary, authors, link, published_at, category, score
                FROM papers
                ORDER BY score DESC, published_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def recent_by_days(self, days: int = 7) -> list[tuple]:
        threshold = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                """
                SELECT source, source_id, title, summary, authors, link, published_at, category, score
                FROM papers
                WHERE COALESCE(published_at, created_at) >= ?
                ORDER BY category ASC, score DESC
                """,
                (threshold,),
            ).fetchall()

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


def detect_category(paper: Paper) -> str:
    text = f"{paper.title} {paper.summary}".lower()
    scores = {cat: sum(1 for kw in kws if kw in text) for cat, kws in CATEGORY_RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())


def text_to_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    for token in tokenize_text(text):
        idx = hash(token) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def build_weighted_profile_vector(keywords: dict[str, float], dim: int = EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    for kw, w in keywords.items():
        kw_vec = text_to_embedding(kw, dim=dim)
        for i in range(dim):
            vec[i] += kw_vec[i] * w
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


def fetch_arxiv(query: str = "all:quantum", max_results: int = 40) -> list[Paper]:
    import feedparser
    import requests

    params = {"search_query": query, "start": 0, "max_results": max_results}
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


def build_digest_markdown(rows: Iterable[tuple]) -> str:
    lines = ["# Personalized Paper Digest", ""]
    for source, source_id, title, summary, authors_json, link, published, category, score in rows:
        try:
            authors = ", ".join(json.loads(authors_json))
        except Exception:
            authors = authors_json
        lines.extend(
            [
                f"## [{title}]({link})",
                f"- Source: {source}",
                f"- Source ID: {source_id}",
                f"- Authors: {authors}",
                f"- Published: {published}",
                f"- Category: {category}",
                f"- Relevance score: {score}",
                f"- Summary: {summary[:320]}..." if summary else "- Summary:",
                "",
            ]
        )
    return "\n".join(lines)


def build_preferences_from_zotero_export(zotero_export_path: str, output_path: str = "preferences.generated.json", top_k_keywords: int = 40) -> None:
    with open(zotero_export_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    corpus: list[str] = []
    authors = Counter()
    tags = Counter()
    for rec in records:
        title = rec.get("title", "")
        abstract = rec.get("abstractNote", "")
        corpus.append(f"{title} {abstract}".lower())
        for c in rec.get("creators", []):
            full_name = " ".join(filter(None, [c.get("firstName", "").strip(), c.get("lastName", "").strip()])).strip()
            if not full_name:
                full_name = c.get("name", "").strip()
            if full_name:
                authors[full_name] += 1
        for tag in rec.get("tags", []):
            tag_value = tag.get("tag", "") if isinstance(tag, dict) else str(tag)
            if tag_value:
                tags[tag_value.lower()] += 1

    prompt = _build_preference_prompt(corpus=corpus, authors=authors, tags=tags, top_k_keywords=top_k_keywords)
    preference = _llm_preference_json(prompt)
    if preference is None:
        preference = _heuristic_preference_json(corpus=corpus, authors=authors, tags=tags, top_k_keywords=top_k_keywords)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preference, f, ensure_ascii=False, indent=2)


def _build_preference_prompt(corpus: list[str], authors: Counter, tags: Counter, top_k_keywords: int) -> str:
    content_samples = "\n".join(f"- {text[:400]}" for text in corpus[:200])
    top_authors = ", ".join(f"{name}({count})" for name, count in authors.most_common(30))
    top_tags = ", ".join(f"{name}({count})" for name, count in tags.most_common(30))
    return (
        "Build a strict JSON preference profile with schema "
        "{\"likes\":[{\"weight\":number,\"keywords\":[string],\"authors\":[string]}]}. "
        f"Use up to {top_k_keywords} concise keywords and 2-6 groups.\n"
        f"Top authors: {top_authors}\n"
        f"Top tags: {top_tags}\n"
        f"Content samples:\n{content_samples}"
    )


def _llm_preference_json(prompt: str) -> dict | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    import requests

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": model,
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
        extracted = _extract_json_object(content)
        return json.loads(extracted) if extracted else None


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _heuristic_preference_json(corpus: list[str], authors: Counter, tags: Counter, top_k_keywords: int) -> dict:
    tokens = tokenize_text(" ".join(corpus))
    stopwords = {
        "with", "from", "that", "this", "their", "have", "using", "into", "results", "study", "analysis",
        "paper", "based", "through", "between", "after", "before", "across", "toward", "show", "shows",
    }
    token_counts = Counter(t for t in tokens if t not in stopwords)
    keywords = [k for k, _ in token_counts.most_common(max(top_k_keywords, 10))]
    boosted = [k for k, _ in tags.most_common(10) if k]
    keywords = list(dict.fromkeys(boosted + keywords))[:top_k_keywords]
    return {
        "likes": [
            {
                "weight": 2.0,
                "keywords": keywords,
                "authors": [name for name, _ in authors.most_common(20)],
            }
        ]
    }


def sync_to_zotero(rows: Iterable[tuple], top_n: int = 10) -> None:
    api_key = os.getenv("ZOTERO_API_KEY")
    user_id = os.getenv("ZOTERO_USER_ID")
    collection_key = os.getenv("ZOTERO_COLLECTION_KEY", "")
    if not api_key or not user_id:
        print("[zotero] missing ZOTERO_API_KEY/ZOTERO_USER_ID; skip sync")
        return

    import requests

    items = []
    for i, (_source, _source_id, title, _summary, authors_json, link, published, category, score) in enumerate(rows):
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

    url = f"https://api.zotero.org/users/{user_id}/items"
    headers = {"Zotero-API-Key": api_key, "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, data=json.dumps(items), timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Zotero sync failed: {resp.status_code} {resp.text}")


def apply_feedback_update(db_path: str, history_path: str, min_occurrence: int = 1) -> None:
    profile_data = load_profile_json(history_path)
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

    blacklist = sorted([kw for kw, c in dislikes.items() if c >= max(min_occurrence, 2)])[:30]
    profile_data["disliked_keywords"] = blacklist

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
    for category, items in sorted(grouped.items()):
        raw_md.append(f"## {category}")
        for source, source_id, title, summary, _authors, link, published, _cat, score in items[:8]:
            raw_md.append(f"- [{title}]({link}) | {source} | {published} | score={score} | id={source_id}")
            raw_md.append(f"  - {summary[:280]}...")
        raw_md.append("")

    report_content = "\n".join(raw_md)
    llm_summary = summarize_report_with_llm(report_content)
    final = report_content + "\n\n## LLM Summary\n\n" + llm_summary

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final)


def summarize_report_with_llm(raw_report_markdown: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY 未配置，已输出结构化周报草稿（无 LLM 总结）。"

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        "请根据以下论文周报草稿，输出中文总结：\n"
        "1) 按分类提炼关键进展；\n"
        "2) 给出跨分类对比；\n"
        "3) 给出下周跟踪建议（5条以内）。\n\n"
        f"{raw_report_markdown}"
    )

    import requests

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You are a concise research analyst."},
                    {"role": "user", "content": prompt},
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
        for row in rows:
            pid, source, source_id, title, summary, authors_json, link, published = row
            try:
                authors = json.loads(authors_json)
            except Exception:
                authors = []
            category = detect_category(Paper(source, source_id, title, summary, authors, link, published))
            conn.execute("UPDATE papers SET category=? WHERE id=?", (category, pid))


def run_pipeline(history_path: str = "preferences.json", db_path: str = "papers.db", subscriptions_path: str | None = None) -> None:
    profile = PreferenceProfile.from_history(history_path)
    store = PaperStore(db_path)

    arxiv_query, arxiv_max_results, feeds = load_subscription_config(subscriptions_path)
    papers = fetch_arxiv(query=arxiv_query, max_results=arxiv_max_results) + fetch_rss_feeds(feeds=feeds)
    for paper in papers:
        category = detect_category(paper)
        score = profile.score(paper)
        store.upsert(paper, category=category, score=score)

    top_rows = store.top_papers(limit=30)
    with open("digest.md", "w", encoding="utf-8") as f:
        f.write(build_digest_markdown(top_rows))
    sync_to_zotero(top_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Literature subscription and recommendation pipeline")
    parser.add_argument("--history", default="preferences.json", help="Preference history JSON")
    parser.add_argument("--db", default="papers.db", help="SQLite path")
    parser.add_argument("--subscriptions", default="subscriptions.json", help="Subscription config JSON path")

    parser.add_argument("--recluster", action="store_true", help="Re-classify existing papers")
    parser.add_argument("--build-preferences-from-zotero", action="store_true", help="Build preferences from Zotero export JSON")
    parser.add_argument("--zotero-export", default="", help="Zotero export JSON file")
    parser.add_argument("--output-preferences", default="preferences.generated.json", help="Output path for generated preferences")

    parser.add_argument("--feedback", action="store_true", help="Record feedback for an article")
    parser.add_argument("--feedback-source", default="", help="Feedback source")
    parser.add_argument("--feedback-source-id", default="", help="Feedback source_id")
    parser.add_argument("--feedback-action", default="", help="like/dislike/save")
    parser.add_argument("--feedback-note", default="", help="Optional feedback note")
    parser.add_argument("--apply-feedback", action="store_true", help="Apply feedback to update preference history")

    parser.add_argument("--weekly-report", action="store_true", help="Generate weekly report markdown")
    parser.add_argument("--report-output", default="weekly_report.md", help="Weekly report output markdown")
    parser.add_argument("--report-days", type=int, default=7, help="Days for weekly report")

    args = parser.parse_args()

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

    if args.feedback:
        if not args.feedback_source or not args.feedback_source_id or not args.feedback_action:
            raise ValueError("--feedback requires --feedback-source --feedback-source-id --feedback-action")
        PaperStore(args.db).add_feedback(
            source=args.feedback_source,
            source_id=args.feedback_source_id,
            action=args.feedback_action,
            note=args.feedback_note,
        )
        return

    if args.apply_feedback:
        apply_feedback_update(db_path=args.db, history_path=args.history)
        return

    if args.weekly_report:
        generate_weekly_report(db_path=args.db, output_path=args.report_output, days=args.report_days)
        return

    subscriptions = args.subscriptions if Path(args.subscriptions).exists() else None
    run_pipeline(history_path=args.history, db_path=args.db, subscriptions_path=subscriptions)


if __name__ == "__main__":
    main()

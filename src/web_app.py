from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict

from src.lit_digest import PaperStore, generate_weekly_report, run_pipeline, sync_to_zotero

PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Literature Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background:#f7f8fa; }
    .toolbar, .filters { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; align-items:center; }
    .card { margin:16px 0; background:#fff; border-radius:10px; padding:12px; box-shadow:0 1px 4px rgba(0,0,0,.08); }
    button { border:0; background:#165dff; color:#fff; padding:8px 12px; border-radius:8px; cursor:pointer; }
    .secondary { background:#4e5969; }
    table { width:100%; border-collapse: collapse; }
    th, td { padding:8px; border-bottom:1px solid #eee; vertical-align: top; text-align:left; }
    .pill { display:inline-block; padding:2px 6px; border-radius:10px; background:#eef3ff; margin-right:4px; font-size:12px; }
    .ok { padding:10px; background:#e8ffea; border:1px solid #6ed17d; border-radius:8px; margin-bottom:10px; }
    .read { color:#1f7a1f; font-weight:600; }
    .unread { color:#b54708; font-weight:600; }
  </style>
</head>
<body>
  <h1>文献更新与筛选面板</h1>
  {% if message %}<div class="ok">{{ message }}</div>{% endif %}

  <form class="toolbar" method="post" action="{{ url_for('update') }}">
    <input type="number" min="1" max="60" name="keep_days" value="{{ keep_days }}" />
    <span>仅保留近N天（默认3天）</span>
    <button type="submit">更新推荐</button>
  </form>

  <form class="filters" method="get" action="{{ url_for('index') }}">
    <label>阅读状态</label>
    <select name="status">
      <option value="all" {% if status=='all' %}selected{% endif %}>全部</option>
      <option value="unread" {% if status=='unread' %}selected{% endif %}>未读</option>
      <option value="read" {% if status=='read' %}selected{% endif %}>已读</option>
    </select>
    <label>标签</label>
    <select name="tag">
      <option value="">全部标签</option>
      {% for t in tags %}<option value="{{t}}" {% if tag==t %}selected{% endif %}>{{t}}</option>{% endfor %}
    </select>
    <button class="secondary" type="submit">筛选</button>
  </form>

  <form method="post" action="{{ url_for('save_selected') }}" id="selectionForm">
    <div class="toolbar">
      <button type="submit">保存勾选</button>
      <button class="secondary" formaction="{{ url_for('mark_read') }}" type="submit">标记为已读</button>
      <button class="secondary" formaction="{{ url_for('mark_unread') }}" type="submit">标记为未读</button>
      <button formaction="{{ url_for('export_selected') }}" type="submit">导出勾选到 Zotero</button>
      <button class="secondary" formaction="{{ url_for('weekly_report') }}" type="submit">生成周报</button>
    </div>

    {% for category, items in grouped.items() %}
      <div class="card">
        <h2>{{ category }} ({{ items|length }})</h2>
        <table>
          <thead>
            <tr><th>选中</th><th>标题</th><th>来源</th><th>相关度</th><th>类型/标签</th><th>阅读状态</th></tr>
          </thead>
          <tbody>
            {% for p in items %}
              <tr>
                <td><input type="checkbox" name="paper_ids" value="{{p['source']}}||{{p['source_id']}}" {% if p['saved'] %}checked{% endif %}></td>
                <td><a href="{{p['link']}}" target="_blank">{{p['title']}}</a><br><small>{{p['summary'][:160]}}...</small></td>
                <td>{{p['source']}}</td>
                <td>{{p['score']}}</td>
                <td>
                  <span class="pill">{{p['article_type']}}</span>
                  {% for t in p['tags'] %}<span class="pill">{{t}}</span>{% endfor %}
                </td>
                <td><span class="{{p['read_status']}}">{{p['read_status']}}</span></td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    {% endfor %}
  </form>
</body>
</html>
"""


def parse_paper_ids(values: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for value in values:
        if "||" in value:
            source, source_id = value.split("||", 1)
            if source and source_id:
                pairs.append((source, source_id))
    return pairs


def create_app():
    try:
        from flask import Flask, redirect, render_template_string, request, url_for
    except ModuleNotFoundError as exc:
        raise RuntimeError("Flask is required for dashboard. Install with: pip install -r requirements.txt") from exc

    app = Flask(__name__)

    def _load_grouped(status: str, tag: str):
        store = PaperStore(os.getenv("LIT_DB_PATH", "papers.db"))
        rows = store.top_papers(limit=300, status=status, tag=tag)
        saved = _saved_map(store)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for source, source_id, title, summary, authors_json, link, published_at, category, score, tags_json, article_type, read_status in rows:
            try:
                tags = json.loads(tags_json or "[]")
            except Exception:
                tags = []
            grouped[category].append(
                {
                    "source": source,
                    "source_id": source_id,
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published_at": published_at,
                    "score": score,
                    "tags": tags,
                    "article_type": article_type or "article",
                    "read_status": read_status or "unread",
                    "saved": saved.get((source, source_id), False),
                }
            )
        return grouped, store.available_tags()

    @app.get("/")
    def index():
        status = request.args.get("status", "all")
        tag = request.args.get("tag", "")
        keep_days = request.args.get("keep_days", os.getenv("LIT_KEEP_DAYS", "3"))
        grouped, tags = _load_grouped(status, tag)
        return render_template_string(PAGE, grouped=grouped, message=request.args.get("msg", ""), tags=tags, status=status, tag=tag, keep_days=keep_days)

    @app.post("/update")
    def update():
        keep_days = int(request.form.get("keep_days", os.getenv("LIT_KEEP_DAYS", "3")))
        run_pipeline(
            history_path=os.getenv("LIT_HISTORY_PATH", "preferences.json"),
            db_path=os.getenv("LIT_DB_PATH", "papers.db"),
            subscriptions_path=(os.getenv("LIT_SUBSCRIPTIONS_PATH", "subscriptions.json") if os.path.exists(os.getenv("LIT_SUBSCRIPTIONS_PATH", "subscriptions.json")) else None),
            keep_days=keep_days,
        )
        return redirect(url_for("index", msg=f"更新完成，仅保留近{keep_days}天", keep_days=keep_days))

    @app.post("/save-selected")
    def save_selected():
        store = PaperStore(os.getenv("LIT_DB_PATH", "papers.db"))
        ids = parse_paper_ids(request.form.getlist("paper_ids"))
        for source, source_id in ids:
            store.add_feedback(source, source_id, "save", "saved from dashboard")
        return redirect(url_for("index", msg=f"已保存 {len(ids)} 篇"))

    @app.post("/mark-read")
    def mark_read():
        store = PaperStore(os.getenv("LIT_DB_PATH", "papers.db"))
        ids = parse_paper_ids(request.form.getlist("paper_ids"))
        store.mark_read_status(ids, "read")
        return redirect(url_for("index", msg=f"已标记已读 {len(ids)} 篇"))

    @app.post("/mark-unread")
    def mark_unread():
        store = PaperStore(os.getenv("LIT_DB_PATH", "papers.db"))
        ids = parse_paper_ids(request.form.getlist("paper_ids"))
        store.mark_read_status(ids, "unread")
        return redirect(url_for("index", msg=f"已标记未读 {len(ids)} 篇"))

    @app.post("/export-selected")
    def export_selected():
        store = PaperStore(os.getenv("LIT_DB_PATH", "papers.db"))
        ids = set(parse_paper_ids(request.form.getlist("paper_ids")))
        rows = [r for r in store.top_papers(limit=400) if (r[0], r[1]) in ids]
        sync_to_zotero(rows, top_n=len(rows))
        return redirect(url_for("index", msg=f"已导出 {len(rows)} 篇到 Zotero"))

    @app.post("/weekly-report")
    def weekly_report():
        generate_weekly_report(
            db_path=os.getenv("LIT_DB_PATH", "papers.db"),
            output_path=os.getenv("LIT_WEEKLY_OUTPUT", "weekly_report.md"),
            days=int(os.getenv("LIT_REPORT_DAYS", "7")),
        )
        return redirect(url_for("index", msg="周报已生成"))

    return app


def _saved_map(store: PaperStore) -> dict[tuple[str, str], bool]:
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute("SELECT source, source_id FROM feedback WHERE action='save'").fetchall()
    return {(source, source_id): True for source, source_id in rows}


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8787")), debug=False)


if __name__ == "__main__":
    main()

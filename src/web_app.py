from __future__ import annotations

import json
import os
from collections import defaultdict

from src.lit_digest import (
    PaperStore,
    generate_weekly_report,
    run_pipeline,
    sync_to_zotero,
)

PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Literature Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background:#f7f8fa; }
    h1 { margin-bottom: 12px; }
    .toolbar { display:flex; gap:8px; flex-wrap: wrap; margin-bottom:16px; }
    button { border:0; background:#165dff; color:#fff; padding:10px 14px; border-radius:8px; cursor:pointer; }
    .secondary { background:#4e5969; }
    .ok { padding:10px; background:#e8ffea; border:1px solid #6ed17d; border-radius:8px; margin-bottom:10px; }
    .category { margin:16px 0; background:#fff; border-radius:10px; padding:12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    table { width:100%; border-collapse: collapse; }
    th, td { text-align:left; padding:8px; border-bottom:1px solid #eee; vertical-align: top; }
    th { background: #fafafa; }
    .score { font-weight: bold; color:#0f7; color:#0a7d25; }
    .summary { color:#4e5969; max-width: 560px; }
    .small { color:#86909c; font-size:12px; }
  </style>
</head>
<body>
  <h1>文献更新与筛选面板</h1>
  {% if message %}<div class="ok">{{ message }}</div>{% endif %}
  <div class="toolbar">
    <form method="post" action="{{ url_for('update') }}">
      <button type="submit">更新推荐（抓取+打分+分类）</button>
    </form>
    <form method="post" action="{{ url_for('weekly_report') }}">
      <button class="secondary" type="submit">生成周报</button>
    </form>
    <form method="post" action="{{ url_for('export_selected') }}" id="exportForm">
      <button type="submit">导出已勾选到 Zotero</button>
    </form>
  </div>

  <form method="post" action="{{ url_for('save_selected') }}" id="saveForm">
    {% for category, items in grouped.items() %}
      <div class="category">
        <h2>{{ category }} <span class="small">({{ items|length }}篇)</span></h2>
        <table>
          <thead>
            <tr>
              <th>保存</th>
              <th>标题</th>
              <th>来源</th>
              <th>相关度</th>
              <th>发布时间</th>
              <th>摘要</th>
            </tr>
          </thead>
          <tbody>
            {% for p in items %}
            <tr>
              <td>
                <input type="checkbox" name="paper_ids" value="{{p['source']}}||{{p['source_id']}}" {% if p['saved'] %}checked{% endif %} />
              </td>
              <td>
                <a href="{{ p['link'] }}" target="_blank">{{ p['title'] }}</a><br/>
                <span class="small">{{ p['source_id'] }}</span>
              </td>
              <td>{{ p['source'] }}</td>
              <td class="score">{{ p['score'] }}</td>
              <td>{{ p['published_at'] }}</td>
              <td class="summary">{{ p['summary'][:180] }}...</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    {% endfor %}
    <button type="submit">保存勾选（本地收藏）</button>
  </form>

  <script>
    function selectedIds() {
      return Array.from(document.querySelectorAll('input[name="paper_ids"]:checked')).map(e => e.value);
    }
    document.getElementById('exportForm').addEventListener('submit', function(e){
      const form = e.target;
      selectedIds().forEach(v => {
        const input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'paper_ids';
        input.value = v;
        form.appendChild(input);
      });
    });
  </script>
</body>
</html>
"""


def parse_paper_ids(values: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        if "||" not in value:
            continue
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
    app.config["SECRET_KEY"] = os.getenv("APP_SECRET_KEY", "local-dev-secret")

    def _load_rows() -> tuple[dict[str, list[dict]], dict[tuple[str, str], bool]]:
        db_path = os.getenv("LIT_DB_PATH", "papers.db")
        store = PaperStore(db_path)
        rows = store.top_papers(limit=200)
        saved_map = _saved_map(store)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for source, source_id, title, summary, authors_json, link, published_at, category, score in rows:
            try:
                authors = ", ".join(json.loads(authors_json))
            except Exception:
                authors = authors_json
            grouped[category].append(
                {
                    "source": source,
                    "source_id": source_id,
                    "title": title,
                    "summary": summary,
                    "authors": authors,
                    "link": link,
                    "published_at": published_at,
                    "category": category,
                    "score": score,
                    "saved": saved_map.get((source, source_id), False),
                }
            )
        for category in grouped:
            grouped[category] = sorted(grouped[category], key=lambda x: x["score"], reverse=True)
        return grouped, saved_map

    @app.get("/")
    def index():
        grouped, _ = _load_rows()
        message = request.args.get("msg", "")
        return render_template_string(PAGE, grouped=grouped, message=message)

    @app.post("/update")
    def update():
        run_pipeline(
            history_path=os.getenv("LIT_HISTORY_PATH", "preferences.json"),
            db_path=os.getenv("LIT_DB_PATH", "papers.db"),
            subscriptions_path=(os.getenv("LIT_SUBSCRIPTIONS_PATH", "subscriptions.json") if os.path.exists(os.getenv("LIT_SUBSCRIPTIONS_PATH", "subscriptions.json")) else None),
        )
        return redirect(url_for("index", msg="更新完成：推荐已刷新"))

    @app.post("/save-selected")
    def save_selected():
        db_path = os.getenv("LIT_DB_PATH", "papers.db")
        store = PaperStore(db_path)
        selected = parse_paper_ids(request.form.getlist("paper_ids"))
        for source, source_id in selected:
            store.add_feedback(source=source, source_id=source_id, action="save", note="saved from dashboard")
        return redirect(url_for("index", msg=f"已保存 {len(selected)} 篇文献"))

    @app.post("/export-selected")
    def export_selected():
        db_path = os.getenv("LIT_DB_PATH", "papers.db")
        store = PaperStore(db_path)
        selected_ids = set(parse_paper_ids(request.form.getlist("paper_ids")))
        rows = store.top_papers(limit=300)
        selected_rows = [r for r in rows if (r[0], r[1]) in selected_ids]
        sync_to_zotero(selected_rows, top_n=len(selected_rows) or 0)
        return redirect(url_for("index", msg=f"已导出 {len(selected_rows)} 篇到 Zotero"))

    @app.post("/weekly-report")
    def weekly_report():
        generate_weekly_report(
            db_path=os.getenv("LIT_DB_PATH", "papers.db"),
            output_path=os.getenv("LIT_WEEKLY_OUTPUT", "weekly_report.md"),
            days=int(os.getenv("LIT_REPORT_DAYS", "7")),
        )
        return redirect(url_for("index", msg="周报已生成（weekly_report.md）"))

    return app


def _saved_map(store: PaperStore) -> dict[tuple[str, str], bool]:
    import sqlite3

    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute("SELECT source, source_id FROM feedback WHERE action='save'").fetchall()
    return {(r[0], r[1]): True for r in rows}


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8787")), debug=False)


if __name__ == "__main__":
    main()

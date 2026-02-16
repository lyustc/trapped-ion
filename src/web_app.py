from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict, OrderedDict

from src.lit_digest import PaperStore, generate_weekly_report, run_pipeline, sync_to_zotero, load_subscription_config

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

  <form class="toolbar" method="post" action="/update">
    <input type="number" min="1" max="60" name="keep_days" value="{{ keep_days }}" />
    <span>仅保留近N天（默认3天）</span>
        <label>arXiv max</label>
        <input type="number" name="arxiv_max_results" min="1" max="500" value="{{ arxiv_max_results }}" style="width:88px" />
    <button type="submit">更新推荐</button>
  </form>
    <script>
        // Collapse/expand category cards and mark clicked links as read
        document.addEventListener('DOMContentLoaded', function(){
            // restore collapsed state from localStorage
            const collapsedKey = 'lit.collapsed.categories';
            let collapsed = [];
            try { collapsed = JSON.parse(localStorage.getItem(collapsedKey) || '[]'); } catch(e) { collapsed = []; }
            document.querySelectorAll('.card > h2').forEach(function(h){
                h.style.cursor = 'pointer';
                const cat = h.textContent.trim().split('\n')[0];
                const tbl = h.nextElementSibling;
                if (collapsed.includes(cat) && tbl) tbl.style.display = 'none';
                h.addEventListener('click', function(){
                    if (!tbl) return;
                    const isHidden = tbl.style.display === 'none';
                    tbl.style.display = isHidden ? '' : 'none';
                    try{
                        if (!isHidden) {
                            // now hidden -> add to collapsed
                            if (!collapsed.includes(cat)) { collapsed.push(cat); }
                        } else {
                            // now shown -> remove
                            collapsed = collapsed.filter(x=>x!==cat);
                        }
                        localStorage.setItem(collapsedKey, JSON.stringify(collapsed));
                    } catch(e) {}
                });
            });
            // mark external links as read when clicked (sendBeacon or fetch)
            document.querySelectorAll('a.external-link').forEach(function(a){
                a.addEventListener('click', function(){
                    const src = a.dataset.source;
                    const sid = a.dataset.sourceId;
                    const payload = JSON.stringify({source: src, source_id: sid});
                    if (navigator.sendBeacon){
                        navigator.sendBeacon('/mark-read-click', payload);
                    } else {
                        fetch('/mark-read-click', {method: 'POST', headers: {'Content-Type':'application/json'}, body: payload}).catch(()=>{});
                    }
                });
            });
        });
    </script>

  <!-- ZOTERO IMPORT START -->
  <div class="card">
    <h3>从 Zotero 导入并更新偏好</h3>
    <form method="post" action="/zotero-upload" enctype="multipart/form-data">
      <label>上传 Zotero 导出 (JSON)</label>
      <input type="file" name="zotero_file" accept="application/json" />
      <label>合并到历史偏好</label>
      <input type="checkbox" name="merge" />
      <button type="submit">处理 Zotero 导出</button>
    </form>
  </div>
  <!-- ZOTERO IMPORT END -->

  <form class="filters" method="get" action="/">
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

  <form method="post" action="/save-selected" id="selectionForm">
    <div class="toolbar">
      <button type="submit">保存勾选</button>
      <button class="secondary" formaction="/mark-read" type="submit">标记为已读</button>
      <button class="secondary" formaction="/mark-unread" type="submit">标记为未读</button>
      <button formaction="/export-selected" type="submit">导出勾选到 Zotero</button>
      <button class="secondary" formaction="/weekly-report" type="submit">生成周报</button>
    </div>

    {% for category in category_order %}
        {% set items = grouped[category] %}
        <div class="card">
            <h2>{{ category }} ({{ items|length }})</h2>
            <table>
                <thead>
                                        <tr><th>选中</th><th>标题</th><th>发布时间</th><th>来源</th><th>相关度</th><th>类型/标签</th><th>阅读状态</th></tr>
                </thead>
                <tbody>
                    {% for item in items %}
                                        <tr>
                                                <td><input type="checkbox" name="paper_ids" value="{{ item.source }}||{{ item.source_id }}" {% if item.saved %}checked{% endif %} /></td>
                                                                                                <td><a class="external-link" data-source="{{ item.source }}" data-source-id="{{ item.source_id }}" href="{{ item.link }}" target="_blank">{{ item.title }}</a><br><small>{{ item.summary|truncate(200) }}</small></td>
                                                                                                <td>{{ item.published_at[:10] }}</td>
                        <td>{{ item.source }}</td>
                        <td>{{ "%.3f"|format(item.score) }}</td>
                        <td>
                            <span class="pill">{{ item.article_type }}</span>
                            {% for tag in item.tags %}<span class="pill">{{ tag }}</span>{% endfor %}
                        </td>
                        <td><span class="{% if item.read_status == 'read' %}read{% else %}unread{% endif %}">{{ item.read_status }}</span></td>
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
        temp_grouped: dict[str, list[dict]] = defaultdict(list)
        category_order: list[str] = []
        for source, source_id, title, summary, authors_json, link, published_at, category, score, tags_json, article_type, read_status in rows:
            try:
                tags = json.loads(tags_json or "[]")
            except Exception:
                tags = []
            if category not in category_order:
                category_order.append(category)
            temp_grouped[category].append(
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
        # Preserve the order determined by top_papers (quantum categories -> quantum-others -> other)
        ordered = OrderedDict()
        for c in category_order:
            ordered[c] = temp_grouped.get(c, [])
        return ordered, store.available_tags()

    @app.get("/")
    def index():
        status = request.args.get("status", "all")
        tag = request.args.get("tag", "")
        keep_days = request.args.get("keep_days", os.getenv("LIT_KEEP_DAYS", "3"))
        grouped, tags = _load_grouped(status, tag)
        # explicit category order to ensure `quantum-others` sits before `other`
        category_order = list(grouped.keys())
        # provide default arXiv max from subscription config
        _, arxiv_default_max, _ = load_subscription_config(None)
        arxiv_max_results = request.args.get("arxiv_max_results", arxiv_default_max)
        return render_template_string(PAGE, grouped=grouped, category_order=category_order, message=request.args.get("msg", ""), tags=tags, status=status, tag=tag, keep_days=keep_days, arxiv_max_results=arxiv_max_results)

    @app.post("/update")
    def update():
        keep_days = int(request.form.get("keep_days", os.getenv("LIT_KEEP_DAYS", "3")))

        # keep dashboard simple: use configured default arXiv query; allow adjusting max results
        arxiv_max = int(request.form.get("arxiv_max_results", 40))
        subs_path = None

        run_pipeline(
            history_path=os.getenv("LIT_HISTORY_PATH", "preferences.json"),
            db_path=os.getenv("LIT_DB_PATH", "papers.db"),
            subscriptions_path=subs_path,
            arxiv_max_results=arxiv_max,
            keep_days=keep_days,
        )
        return redirect(url_for("index", msg=f"更新完成，仅保留近{keep_days}天", keep_days=keep_days))

    @app.post('/zotero-upload')
    def zotero_upload():
        # handle uploaded zotero JSON and either build a preview or merge into history
        f = request.files.get('zotero_file')
        if not f:
            return redirect(url_for('index', msg='未提供 Zotero 导出文件'))
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json', dir='.')
        try:
            f.save(tmp.name)
            tmp.close()
            merge = bool(request.form.get('merge'))
            # If merge checked, update preferences in place; otherwise build preview file
            out_path = 'preferences.merged.json' if not merge else os.getenv('LIT_HISTORY_PATH', 'preferences.json')
            if merge:
                # merge into history
                try:
                    from src.lit_digest import update_preferences_from_zotero_export

                    update_preferences_from_zotero_export(tmp.name, history_path=out_path, top_k_keywords=100)
                except Exception as e:
                    return redirect(url_for('index', msg=f'合并失败: {e}'))
                return redirect(url_for('index', msg=f'偏好已合并到 {out_path}'))
            else:
                try:
                    from src.lit_digest import build_preferences_from_zotero_export

                    build_preferences_from_zotero_export(tmp.name, output_path=out_path, top_k_keywords=100)
                except Exception as e:
                    return redirect(url_for('index', msg=f'生成预览失败: {e}'))
                return redirect(url_for('index', msg=f'已生成 {out_path} 供审查'))
        finally:
            try:
                os.remove(tmp.name)
            except Exception:
                pass

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

    @app.post('/mark-read-click')
    def mark_read_click():
        try:
            data = request.get_json(force=True)
            src = data.get('source')
            sid = data.get('source_id')
            if src and sid:
                store = PaperStore(os.getenv("LIT_DB_PATH", "papers.db"))
                store.mark_read_status([(src, sid)], 'read')
        except Exception:
            pass
        return ('', 204)

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

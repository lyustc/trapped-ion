from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone

from src.lit_digest import PaperStore, generate_weekly_report, run_pipeline

DISPLAY_CATEGORY_ORDER = [
    "trapped-ion",
    "quantum-information",
    "quantum-computing",
    "quantum-metrology",
    "quantum-simulation",
    "quantum-platform",
    "quantum-others",
    "other",
]

PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Literature Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background:#f7f8fa; }
    .page-layout { display:block; }
    .nav-toggle { position:fixed; top:12px; left:12px; z-index:1100; border:0; background:#165dff; color:#fff; padding:8px 10px; border-radius:8px; cursor:pointer; box-shadow:0 2px 10px rgba(0,0,0,.15); }
    .sidebar-backdrop { position:fixed; inset:0; background:rgba(15,23,42,.28); z-index:1040; display:none; }
    .sidebar-backdrop.open { display:block; }
    .sidebar { width:240px; position:fixed; left:12px; top:52px; bottom:12px; z-index:1050; overflow:auto; background:#fff; border-radius:10px; padding:12px; box-shadow:0 8px 24px rgba(0,0,0,.18); transform:translateX(-120%); transition:transform .18s ease; }
    .sidebar.open { transform:translateX(0); }
    .sidebar h3 { margin:0 0 8px; font-size:15px; }
    .side-link { display:block; text-decoration:none; color:#1f2937; padding:6px 8px; border-radius:8px; margin:2px 0; }
    .side-link:hover { background:#eef3ff; color:#165dff; }
    .main-content { min-width:0; }
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
    .score-wrap { min-width: 160px; }
    .score-bg { width: 120px; height: 10px; background: #eef1f5; border-radius: 999px; overflow: hidden; display: inline-block; vertical-align: middle; margin-right: 8px; }
    .score-fill { display:block; height: 100%; border-radius: 999px; }
    .score-high { background: linear-gradient(90deg, #16a34a, #22c55e); }
    .score-mid { background: linear-gradient(90deg, #eab308, #f59e0b); }
    .score-low { background: linear-gradient(90deg, #ef4444, #f97316); }
    .score-text { color: #334155; font-size: 12px; }
    @media (max-width: 980px) {
      .sidebar { width:min(86vw, 320px); left:8px; top:48px; bottom:8px; }
      .nav-toggle { top:8px; left:8px; padding:7px 9px; }
    }
  </style>
</head>
<body>
  <button id="navToggle" class="nav-toggle" type="button">分类</button>
  <div id="sidebarBackdrop" class="sidebar-backdrop"></div>
  <div class="page-layout">
    <aside id="sidebarNav" class="sidebar">
      <h3>分类导航</h3>
      {% for category in category_order %}
      <a class="side-link" href="#cat-{{ category|replace(' ', '-') }}">{{ category }} ({{ grouped[category]|length }})</a>
      {% endfor %}
    </aside>
    <main class="main-content">
  <h1>文献更新与筛选面板</h1>
  {% if message %}<div class="ok">{{ message }}</div>{% endif %}

  <form class="toolbar" method="post" action="/update">
    <input type="number" min="1" max="60" name="keep_days" value="{{ keep_days }}" />
    <span>仅保留近 N 天（默认 3 天）</span>
    <label><input type="checkbox" name="quick" value="1" /> 快速模式（仅抓最新 arXiv）</label>
    <label><input type="checkbox" name="doi_author" value="1" {% if doi_author == "1" %}checked{% endif %}/> DOI补作者</label>
    <label><input type="checkbox" name="doi_abs" value="1" {% if doi_abs == "1" %}checked{% endif %}/> DOI补摘要</label>
    <label>最长等待秒数</label>
    <input type="number" name="max_runtime_sec" min="20" max="900" value="{{ max_runtime_sec }}" style="width:88px" />
    <button type="submit">更新推荐</button>
  </form>
    <script>
        // Collapse/expand category cards and mark clicked links as read
        document.addEventListener('DOMContentLoaded', function(){
            const navToggle = document.getElementById('navToggle');
            const sidebar = document.getElementById('sidebarNav');
            const backdrop = document.getElementById('sidebarBackdrop');
            const setSidebarOpen = function(open){
                if (!sidebar || !backdrop) return;
                sidebar.classList.toggle('open', !!open);
                backdrop.classList.toggle('open', !!open);
                if (navToggle) navToggle.textContent = open ? '收起分类' : '分类';
            };
            if (navToggle) {
                navToggle.addEventListener('click', function(){
                    const isOpen = sidebar && sidebar.classList.contains('open');
                    setSidebarOpen(!isOpen);
                });
            }
            if (backdrop) {
                backdrop.addEventListener('click', function(){ setSidebarOpen(false); });
            }
            document.querySelectorAll('.side-link').forEach(function(a){
                a.addEventListener('click', function(){ setSidebarOpen(false); });
            });
            // restore collapsed state from localStorage
            const collapsedKey = 'lit.collapsed.categories';
            let collapsed = [];
            try { collapsed = JSON.parse(localStorage.getItem(collapsedKey) || '[]'); } catch(e) { collapsed = []; }
            document.querySelectorAll('.card > h2').forEach(function(h){
                h.style.cursor = 'pointer';
                const cat = h.textContent.replace(/\\(\\d+\\)\\s*$/, '').trim();
                const card = h.closest('.card');
                const tbl = card ? card.querySelector('table') : null;
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
      <label>上传 Zotero 导出文件 (JSON)</label>
      <input type="file" name="zotero_file" accept="application/json" />
      <label>覆盖当前偏好（自动更新 preferences）</label>
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
    <label>arXiv</label>
    <select name="show_arxiv">
      <option value="1" {% if show_arxiv=='1' %}selected{% endif %}>显示</option>
      <option value="0" {% if show_arxiv=='0' %}selected{% endif %}>隐藏</option>
    </select>
    <label>标签</label>
    <select name="tag">
      <option value="">全部标签</option>
      {% for t in tags %}<option value="{{t}}" {% if tag==t %}selected{% endif %}>{{t}}</option>{% endfor %}
    </select>
    <label>排序</label>
    <select name="sort_by">
      <option value="score" {% if sort_by=='score' %}selected{% endif %}>按相关度</option>
      <option value="time" {% if sort_by=='time' %}selected{% endif %}>按时间</option>
    </select>
    <button class="secondary" type="submit">筛选</button>
  </form>

  <form method="post" action="/save-selected" id="selectionForm">
    <input type="hidden" name="status" value="{{ status }}" />
    <input type="hidden" name="tag" value="{{ tag }}" />
    <input type="hidden" name="show_arxiv" value="{{ show_arxiv }}" />
    <input type="hidden" name="sort_by" value="{{ sort_by }}" />
    <div class="toolbar">
      <button type="submit">保存勾选</button>
      <button class="secondary" formaction="/mark-all-read" type="submit">一键全部已读</button>
      <button class="secondary" formaction="/mark-read" type="submit">标记为已读</button>
      <button class="secondary" formaction="/mark-unread" type="submit">标记为未读</button>
      <button formaction="/export-selected" type="submit">导出勾选为 BibTeX (.bib)</button>
      <button class="secondary" formaction="/weekly-report" type="submit">生成周报</button>
    </div>

    {% for category in category_order %}
        {% set items = grouped[category] %}
        <div class="card" id="cat-{{ category|replace(' ', '-') }}">
            <h2>{{ category }} ({{ items|length }})</h2>
            <table>
                <thead>
                    <tr><th>选中</th><th>标题</th><th>发布时间</th><th>来源</th><th>相关度</th><th>类型/标签</th><th>阅读状态</th></tr>
                </thead>
                <tbody>
                    {% for item in items %}
                                        <tr>
                                                <td><input type="checkbox" name="paper_ids" value="{{ item.source }}||{{ item.source_id }}" {% if item.saved %}checked{% endif %} /></td>
                                                                                                <td><a class="external-link" data-source="{{ item.source }}" data-source-id="{{ item.source_id }}" href="{{ item.link }}" target="_blank">{{ item.title }}</a>{% if item.authors_text %}<br><small><strong>作者：</strong>{{ item.authors_text }}</small>{% endif %}<br><small>{{ item.summary|truncate(200) }}</small></td>
                                                                                                <td>{{ item.published_at[:10] }}</td>
                        <td>{{ item.source }}</td>
                        <td class="score-wrap">
                            <span class="score-bg">
                                <span class="score-fill {{ item.score_level }}" style="width: {{ '%.2f'|format(item.score_pct) }}%;"></span>
                            </span>
                            <span class="score-text">{{ "%.2f"|format(item.score) }}/50</span>
                        </td>
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
    </main>
  </div>
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


def _compact_authors(authors: list[str], head: int = 3, tail: int = 2, full_limit: int = 10) -> str:
    clean = [a.strip() for a in (authors or []) if str(a).strip()]
    n = len(clean)
    if n == 0:
        return ""
    if n <= full_limit:
        return ", ".join(clean)
    return ", ".join(clean[:head]) + " ... " + ", ".join(clean[-tail:])


def _bibtex_escape(text: str) -> str:
    if not text:
        return ""
    return str(text).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _rows_to_bibtex(rows: list[tuple]) -> str:
    entries: list[str] = []
    for idx, r in enumerate(rows, start=1):
        source, source_id, title, summary, authors_json, link, published, category, score = r[:9]
        try:
            authors = json.loads(authors_json or "[]")
            if not isinstance(authors, list):
                authors = [str(authors)]
        except Exception:
            authors = [str(authors_json)] if authors_json else []
        author_field = " and ".join(_bibtex_escape(a) for a in authors if a) or "Unknown"
        year = ""
        if published:
            y = str(published)[:4]
            year = y if y.isdigit() else ""
        key = f"{(source or 'paper').replace(' ', '-')}-{idx}"
        journal = _bibtex_escape(source or "unknown")
        doi = ""
        if link and "doi.org/" in link:
            doi = link.split("doi.org/", 1)[1].strip()
        fields = [
            f"  title = {{{_bibtex_escape(title)}}}",
            f"  author = {{{author_field}}}",
            f"  journal = {{{journal}}}",
            f"  url = {{{_bibtex_escape(link or '')}}}",
            f"  note = {{{_bibtex_escape('category=' + str(category) + ', score=' + str(score))}}}",
        ]
        if year:
            fields.append(f"  year = {{{year}}}")
        if doi:
            fields.append(f"  doi = {{{_bibtex_escape(doi)}}}")
        if summary:
            fields.append(f"  abstract = {{{_bibtex_escape(summary)}}}")
        fields.append(f"  annote = {{{_bibtex_escape('source_id=' + str(source_id))}}}")
        entry = "@article{" + key + ",\n" + ",\n".join(fields) + "\n}\n"
        entries.append(entry)
    return "\n".join(entries)


def create_app():
    try:
        from flask import Flask, Response, redirect, render_template_string, request, url_for
    except ModuleNotFoundError as exc:
        raise RuntimeError("Flask is required for dashboard. Install with: pip install -r requirements.txt") from exc

    app = Flask(__name__)

    def _resolve_subscriptions_path() -> str | None:
        cand = os.getenv("LIT_SUBSCRIPTIONS_PATH", "subscriptions.json")
        return cand if os.path.exists(cand) else None

    def _load_grouped(status: str, tag: str, show_arxiv: str, sort_by: str):
        store = PaperStore(os.getenv("LIT_DB_PATH", "papers.db"))
        # Use a high limit so category buckets are not accidentally shown as empty
        # due to truncation from top-N ordering.
        rows = store.top_papers(limit=2000, status=status, tag=tag, include_arxiv=(show_arxiv != "0"), sort_by=sort_by)
        saved = _saved_map(store)
        temp_grouped: dict[str, list[dict]] = defaultdict(list)
        for source, source_id, title, summary, authors_json, link, published_at, category, score, tags_json, article_type, read_status in rows:
            try:
                tags = json.loads(tags_json or "[]")
            except Exception:
                tags = []
            try:
                authors = json.loads(authors_json or "[]")
                if not isinstance(authors, list):
                    authors = [str(authors)]
            except Exception:
                authors = [str(authors_json)] if authors_json else []
            authors_text = _compact_authors(authors, head=3, tail=2)
            score_norm = max(0.0, min(float(score), 50.0))
            score_pct = (score_norm / 50.0) * 100.0
            published_sort = 0.0
            try:
                dt = datetime.fromisoformat((published_at or "").replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                published_sort = dt.timestamp()
            except Exception:
                pass
            if score_pct >= 70:
                score_level = "score-high"
            elif score_pct >= 35:
                score_level = "score-mid"
            else:
                score_level = "score-low"
            temp_grouped[category].append(
                {
                    "source": source,
                    "source_id": source_id,
                    "title": title,
                    "authors_text": authors_text,
                    "summary": summary,
                    "link": link,
                    "published_at": published_at,
                    "published_sort": published_sort,
                    "score": score,
                    "score_pct": score_pct,
                    "score_level": score_level,
                    "tags": tags,
                    "article_type": article_type or "article",
                    "read_status": read_status or "unread",
                    "saved": saved.get((source, source_id), False),
                }
            )
        # Keep UI category order stable regardless of query sorting.
        # Unknown categories are appended before "other"; "other" is always last.
        ordered = OrderedDict()
        present = set(temp_grouped.keys())
        for c in DISPLAY_CATEGORY_ORDER:
            ordered[c] = temp_grouped.get(c, [])
        extras = [c for c in temp_grouped.keys() if c not in DISPLAY_CATEGORY_ORDER and c != "other"]
        for c in sorted(extras):
            ordered[c] = temp_grouped.get(c, [])
        if "other" not in ordered:
            ordered["other"] = temp_grouped.get("other", [])
        return ordered, store.available_tags()

    @app.get("/")
    def index():
        status = request.args.get("status", "all")
        tag = request.args.get("tag", "")
        show_arxiv = request.args.get("show_arxiv", "1")
        sort_by = request.args.get("sort_by", "score")
        keep_days = request.args.get("keep_days", os.getenv("LIT_KEEP_DAYS", "3"))
        max_runtime_sec = request.args.get("max_runtime_sec", os.getenv("LIT_MAX_RUNTIME_SEC", "120"))
        doi_author = request.args.get("doi_author", os.getenv("LIT_ENABLE_DOI_AUTHOR", "1"))
        doi_abs = request.args.get("doi_abs", os.getenv("LIT_ENABLE_DOI_ABS", "0"))
        grouped, tags = _load_grouped(status, tag, show_arxiv, sort_by)
        # explicit category order to ensure `quantum-others` sits before `other`
        category_order = list(grouped.keys())
        return render_template_string(PAGE, grouped=grouped, category_order=category_order, message=request.args.get("msg", ""), tags=tags, status=status, tag=tag, show_arxiv=show_arxiv, sort_by=sort_by, keep_days=keep_days, max_runtime_sec=max_runtime_sec, doi_author=doi_author, doi_abs=doi_abs)

    @app.post("/update")
    def update():
        keep_days = int(request.form.get("keep_days", os.getenv("LIT_KEEP_DAYS", "3")))

        quick = request.form.get("quick", "0") == "1"
        max_runtime_sec = int(request.form.get("max_runtime_sec", os.getenv("LIT_MAX_RUNTIME_SEC", "120")))
        doi_author = request.form.get("doi_author", os.getenv("LIT_ENABLE_DOI_AUTHOR", "1")) == "1"
        doi_abs = request.form.get("doi_abs", os.getenv("LIT_ENABLE_DOI_ABS", "0")) == "1"
        subs_path = _resolve_subscriptions_path()

        result = run_pipeline(
            history_path=os.getenv("LIT_HISTORY_PATH", "preferences.json"),
            db_path=os.getenv("LIT_DB_PATH", "papers.db"),
            subscriptions_path=subs_path,
            keep_days=keep_days,
            quick=quick,
            max_runtime_sec=max_runtime_sec,
            enable_doi_author=doi_author,
            enable_doi_abs=doi_abs,
        )
        if result.get("doi_cutoff_triggered"):
            msg = (
                f"达到 {max_runtime_sec} 秒后已自动关闭 DOI 补全并继续抓取全文献"
                f"（arXiv {result.get('arxiv_count', 0)} 篇，RSS {result.get('rss_count', 0)} 篇，"
                f"总耗时 {result.get('total_sec', '?')} 秒）"
            )
        else:
            msg = (
                f"更新完成，仅保留近 {keep_days} 天（耗时 {result.get('total_sec', '?')} 秒，"
                f"arXiv {result.get('arxiv_count', 0)} 篇，RSS {result.get('rss_count', 0)} 篇）"
            )
        return redirect(url_for("index", msg=msg, keep_days=keep_days, max_runtime_sec=max_runtime_sec, doi_author="1" if doi_author else "0", doi_abs="1" if doi_abs else "0"))

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
            # If merge checked, overwrite preferences in place; otherwise build preview file
            out_path = 'preferences.merged.json' if not merge else os.getenv('LIT_HISTORY_PATH', 'preferences.json')
            if merge:
                # overwrite history
                try:
                    from src.lit_digest import update_preferences_from_zotero_export

                    update_preferences_from_zotero_export(tmp.name, history_path=out_path, top_k_keywords=100)
                except Exception as e:
                    return redirect(url_for('index', msg=f'覆盖失败: {e}'))
                return redirect(url_for('index', msg=f'偏好已覆盖写入 {out_path}'))
            else:
                try:
                    from src.lit_digest import build_preferences_from_zotero_export

                    build_preferences_from_zotero_export(tmp.name, output_path=out_path, top_k_keywords=100)
                except Exception as e:
                    return redirect(url_for('index', msg=f'生成预览失败: {e}'))
                return redirect(url_for('index', msg=f'已生成 {out_path} 供审阅'))
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

    @app.post("/mark-all-read")
    def mark_all_read():
        store = PaperStore(os.getenv("LIT_DB_PATH", "papers.db"))
        status = request.form.get("status", "all")
        tag = request.form.get("tag", "")
        show_arxiv = request.form.get("show_arxiv", "1")
        sort_by = request.form.get("sort_by", "score")
        rows = store.top_papers(
            limit=300,
            status=status,
            tag=tag,
            include_arxiv=(show_arxiv != "0"),
            sort_by=sort_by,
        )
        ids = [(r[0], r[1]) for r in rows]
        store.mark_read_status(ids, "read")
        return redirect(
            url_for(
                "index",
                msg=f"已一键标记已读 {len(ids)} 篇",
                status=status,
                tag=tag,
                show_arxiv=show_arxiv,
                sort_by=sort_by,
            )
        )

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
        if not rows:
            return redirect(url_for("index", msg="未选择任何论文"))
        bib = _rows_to_bibtex(rows)
        filename = f"papers_selected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bib"
        return Response(
            bib,
            mimetype="application/x-bibtex; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
        )

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


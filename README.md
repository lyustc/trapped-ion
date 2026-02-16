# 文献摘要与推荐工具 — 完整说明

这是一个本地运行的科研文献跟踪与推荐工具，面向量子与相关物理方向。工具从 arXiv 与若干 RSS 源抓取条目，结合用户偏好打分并在网页界面上展示，支持交互式标记与偏好更新。

## 主要功能

- 数据源：自动抓取 arXiv（默认量子/atomic-physics 查询）与 RSS（如 Nature / Science / PRL）。
- 分类与过滤：基于规则与短语触发器进行类别检测，区分 `trapped-ion` 与 `quantum-platform` 等量子相关类别；非量子噪声默认归入 `other`。
- 偏好驱动推荐：从 `preferences.json` 加载关键词与作者偏好，按偏好打分并排序展示。
- Zotero 支持：通过上传 Zotero 导出的 JSON 生成/合并偏好（作者名标准化与去重）。
- Web 仪表盘：启动 Flask 服务器在浏览器查看结果，支持折叠类目、按点击标记为已读、显示发表时间、调整 `arxiv_max_results` 与 `keep_days`。
- 数据库维护：`recluster_existing()` 重新分类现有记录，`clean_database()` 清理非量子噪声并保留高影响期刊与物理相关条目。
- 同步：可选择将推荐条目同步回 Zotero（`sync_to_zotero`）。

## 快速开始

1. 创建并激活虚拟环境（推荐）

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. 安装依赖

```powershell
pip install -r requirements.txt
```

3. 配置（可选）

- 复制 `preferences.example.json` 为 `preferences.json` 并根据需要修改关键词/作者权重。
- 如果需要自定义 arXiv 查询或 RSS 源，复制 `subscriptions.example.json` 并编辑 `arxiv_query` / `arxiv_max_results` / `rss_feeds`。

## 运行与常用命令

- 运行推荐管线并生成本地 `digest.md`：

```powershell
python -m src.lit_digest --db papers.db --keep-days 3
```

- 仅重分类现有数据库（不抓取）：

```powershell
python -m src.lit_digest --recluster --db papers.db
```

- 清理数据库（移除非量子噪声）：

```powershell
python -c "from src.lit_digest import clean_database; print(clean_database('papers.db', keep_days=3))"
```

- 启动网页仪表盘：

```powershell
python src/web_app.py
```

在浏览器打开 http://127.0.0.1:8787

## 网页界面功能

- 顶部工具条：可以设置 `keep_days`（保留的近期天数）与 `arxiv_max_results`（arXiv 拉取上限）。
- 列表展示：显示论文标题、作者、发表时间、类别与匹配分数。
- 折叠类目卡片：每个类别可以折叠与展开，折叠状态保存在 `localStorage`。
- 点击外部链接时自动 POST 到 `/mark-read-click`，将对应论文标记为已读。

## 分类规则与策略

- 采用显式触发词（如 `trapped ion` / `ion trap`）优先判定 `trapped-ion` 类别，使用词边界匹配避免子串误判。
- 其它物理系统（例如超导原件、量子点、原子等）聚合到 `quantum-platform` 类别。
- 只保留与量子显著相关的条目。

## 数据库与 API（开发者）

- `src/lit_digest.py`:
  - `run_pipeline(...)`: 抓取 arXiv/RSS、检测类别、打分并写入 `papers.db`。
  - `recluster_existing(db_path)`: 重新对 `papers` 表中条目分类并更新 `tags`/`article_type`。
  - `clean_database(db_path, keep_days)`: 重新分类并删除非量子/非物理且非允许期刊的条目。
  - `build_preferences_from_zotero(...)` / `update_preferences_from_zotero(...)`: 从 Zotero 导出构建或合并偏好。
  - `sync_to_zotero(rows)`: 将推荐条目批量推送回 Zotero（需要环境变量 `ZOTERO_API_KEY` / `ZOTERO_USER_ID`）。
- `src/web_app.py`:
  - Flask 应用负责渲染仪表盘、处理 `/mark-read-click`（标记已读）等交互。

## 测试与维护

- 单元测试：运行 `pytest`（项目包含 `tests/test_lit_digest.py`）。

```powershell
pytest -q
```

- 若出现分类误判，可编辑 `src/lit_digest.py` 中的触发词列表 `TRAPPED_ION_TRIGGERS` / `QUANTUM_PLATFORM_TRIGGERS` 或 `CATEGORY_RULES`，并运行 `--recluster` 与 `clean_database`。

## 开发者备注

- 数据库文件：`papers.db`（SQLite）存储论文条目与 `feedback`。
- 偏好历史：`preferences.json` 用于保存用户偏好。
- 默认抓取策略：arXiv 默认按 `submittedDate` 降序抓取，`arxiv_max_results` 默认为配置文件中的值（网页上可调整）。

---

Generated on 2026-02-16

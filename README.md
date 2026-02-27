# 文献跟踪与推荐工具（Trapped-Ion 优化版）

这是一个本地运行的科研文献聚合与筛选工具，当前默认针对 **Quantum Trapped-Ion** 方向做了规则和来源优化。

工具会从 arXiv + 多个期刊 RSS 抓取条目，进行分类、打分、过滤，并通过 Web 页面进行浏览与交互。

## 1. 主要功能

- 多源抓取：
  - arXiv（可配置 query 与最大条数）
  - RSS（Nature / Science / APS 等）
- 自动分类：
  - 重点分类 `trapped-ion`、`quantum-*`、`other`
  - 分类顺序固定，`other` 永远在最后
- 近期期刊优先：
  - 列表排序会优先已发表来源（非 arXiv）
- 交互式看板：
  - 按分类折叠/展开
  - 过滤已读/未读、标签、是否显示 arXiv
  - 一键更新与重抓
- 阅读与导出：
  - 点击论文自动标记已读，刷新网页后更新
  - 支持勾选后导出为 BibTeX 文件（`.bib`），再导入 Zotero
- 偏好支持：
  - 通过 `preferences.json` 维护关键词/作者偏好用于打分
  - 支持导入 Zotero 导出的 JSON，并覆盖更新 `preferences.json`
  - 关键词自动去重归一，固定保留 100 个并按 20/80 分层权重（2/1）
  - 作者按出现次数筛选（>10 保留，>20 权重 2）

## 2. 快速开始

### 2.1 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2.2 配置文件

- 订阅配置：`subscriptions.json`
  - `arxiv_query`
  - `arxiv_max_results`
  - `rss_feeds`
- 偏好配置：`preferences.json`

如果没有可先参考：
- `subscriptions.example.json`
- `preferences.example.json`

### 2.3 启动看板

```powershell
python src/web_app.py
```

浏览器打开：

```text
http://127.0.0.1:8787
```

### 2.4 Zotero JSON 导入并覆盖更新 preferences

在看板页面的“从 Zotero 导入并更新偏好”模块中：

1. 上传 Zotero 导出的 JSON 文件  
2. 勾选“覆盖当前偏好（自动更新 preferences）”  
3. 提交后会自动把抽取到的作者/关键词覆盖写入 `preferences.json`

如果不勾选覆盖选项，会生成预览文件（默认 `preferences.merged.json`）供你先审阅。

## 3. 常用命令

### 3.1 命令行跑一次抓取/更新

```powershell
python -m src.lit_digest --db papers.db --subscriptions subscriptions.json --keep-days 7
```

### 3.2 仅重分类已有数据

```powershell
python -m src.lit_digest --recluster --db papers.db
```

### 3.3 生成周报

```powershell
python -m src.lit_digest --weekly-report --db papers.db --report-days 7
```

### 3.4 运行测试

```powershell
pytest -q
```

### 3.5 导出选中文献为 BibTeX

在 Web 页面勾选条目后点击“导出勾选为 BibTeX (.bib)”，浏览器会下载 `.bib` 文件。  
你可以在 Zotero 中使用 `File -> Import` 导入该文件。

## 4. 当前针对 Trapped-Ion 的优化点

当前代码中对 trapped-ion 方向做了定向优化，主要在：

- 触发词与分类规则（`src/lit_digest.py`）
  - `TRAPPED_ION_TRIGGERS`
  - `QUANTUM_TRIGGERS`
  - `CATEGORY_RULES`
- “other” 过滤策略（优先保留量子相关）
- 期刊源选择与保留逻辑
- 页面展示顺序（固定分类顺序，`other` 在最后）

## 5. 如果改为其它研究方向，如何调整

假设你要改成例如 `superconducting qubit`、`quantum networking`、`materials` 等方向，建议按下面顺序改：

### 5.1 改数据源

编辑 `subscriptions.json`：

- 修改 `arxiv_query` 为目标领域关键词
- 增删 `rss_feeds`（保留高相关来源，去掉噪声来源）

### 5.2 改分类规则

编辑 `src/lit_digest.py`：

- 替换/新增 `CATEGORY_RULES` 的类别和关键词
- 替换 `TRAPPED_ION_TRIGGERS` 为新方向触发词
- 视需求调整 `QUANTUM_TRIGGERS`（或领域通用触发词）

### 5.3 改过滤与保留策略

仍在 `src/lit_digest.py`：

- 调整 `allowed_other_journals`
- 调整 `run_pipeline()` 里 `other` 的保留逻辑
- 如需更严格，增加“必须包含某些关键词”条件

### 5.4 改前端展示顺序

编辑 `src/web_app.py`：

- 修改 `DISPLAY_CATEGORY_ORDER`，使新分类顺序符合你的阅读习惯


## 6. 目录说明

- `src/lit_digest.py`：抓取、分类、打分、入库、报告
- `src/web_app.py`：Flask 看板与交互
- `papers.db`：SQLite 数据库
- `subscriptions.json`：数据源配置
- `preferences.json`：偏好配置
- `tests/`：测试

---

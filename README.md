# trapped-ion

一个用于**订阅 + 初筛 + 分类 + 存储 + Zotero 集成**的科研文献流水线。

## 已实现能力

- 多源订阅：arXiv + Nature/Nature Physics/Nature Photonics/Nature Communications + Science/Science Advances + PRL/PR Applied/PRX/PRX Quantum。
- 手动可配订阅范围（`subscriptions.json`）。
- 相关度排序：关键词+作者+向量召回（embedding）混合打分（已不再是纯关键词匹配）。
- SQLite 存储、`digest.md` 生成、Zotero 高分同步。
- 从 Zotero 导出库自动生成偏好（LLM优先，失败自动回退启发式）。
- 已读反馈回流：记录 like/dislike/save，自动更新偏好画像。
- 周报能力：按分类输出 weekly report，并可附加 LLM 总结/对比分析。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp preferences.example.json preferences.json
cp subscriptions.example.json subscriptions.json
python -m src.lit_digest --history preferences.json --subscriptions subscriptions.json
```

## 1) 定时任务：每天跑一次

### GitHub Actions（已内置）

仓库已提供 `.github/workflows/daily-digest.yml`，默认 UTC 01:00 每天执行，产出：

- `digest.md`
- `weekly_report.md`
- `papers.db`

需要在仓库 Secrets 中配置（可选）：

- `OPENAI_API_KEY`
- `ZOTERO_API_KEY`
- `ZOTERO_USER_ID`
- `ZOTERO_COLLECTION_KEY`

### crontab（本地服务器）

```bash
crontab -e
# 每天 09:00
0 9 * * * /workspace/trapped-ion/scripts/run_daily.sh >> /workspace/trapped-ion/daily.log 2>&1
```

## 2) 向量召回（embedding）

当前已内置轻量 embedding（哈希向量 + 余弦相似度）并并入总分，主排序不再仅依赖关键词。

可在代码中查看：`text_to_embedding`、`cosine_similarity`、`PreferenceProfile.score`。

## 3) 已读反馈回流，动态更新偏好

### 记录反馈

```bash
python -m src.lit_digest --feedback \
  --db papers.db \
  --feedback-source arxiv \
  --feedback-source-id http://arxiv.org/abs/xxxx \
  --feedback-action like \
  --feedback-note "very relevant"
```

`--feedback-action` 支持：`like` / `dislike` / `save`。

### 应用反馈到偏好文件

```bash
python -m src.lit_digest --apply-feedback --db papers.db --history preferences.json
```

会更新：

- `likes[0].keywords`
- `likes[0].authors`
- `disliked_keywords`

## 4) LLM 自动摘要 + 分类对比周报

```bash
python -m src.lit_digest --weekly-report --db papers.db --report-output weekly_report.md --report-days 7
```

- 有 `OPENAI_API_KEY`：输出含 LLM 中文总结（按分类进展 + 跨分类对比 + 下周建议）。
- 无 Key：仍输出结构化周报草稿。

## 订阅范围配置

编辑 `subscriptions.json`：

```json
{
  "arxiv_query": "all:quantum",
  "arxiv_max_results": 40,
  "rss_feeds": {
    "prx-quantum": "https://journals.aps.org/rss/recent/prxquantum.xml"
  }
}
```

## 从 Zotero 现有文献生成偏好（可再手工修改）

```bash
python -m src.lit_digest \
  --build-preferences-from-zotero \
  --zotero-export zotero-export.json \
  --output-preferences preferences.generated.json
```

## 常用命令

```bash
# 主流程
python -m src.lit_digest --history preferences.json --subscriptions subscriptions.json

# 重分类
python -m src.lit_digest --recluster --db papers.db

# 周报
python -m src.lit_digest --weekly-report --db papers.db
```

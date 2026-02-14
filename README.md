# trapped-ion

一个用于**订阅 + 初筛 + 分类 + 存储 + Zotero 集成预留**的科研文献流水线（MVP）。

## 功能概览

- 订阅来源：
  - arXiv（默认查询 `all:quantum`）
  - Nature RSS
  - Science RSS
  - PRL RSS
- 根据历史偏好（关键词、作者、权重）计算相关度分数。
- 自动分类（quantum / ml-ai / materials / bio / astro / other）。
- 结果写入 SQLite（`papers.db`），并生成 `digest.md` 推送稿。
- 支持定期“重梳理/重分类”（recluster）。
- 支持将高分论文自动同步到 Zotero（通过 Zotero Web API）。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp preferences.example.json preferences.json
python -m src.lit_digest
```

运行后会得到：

- `papers.db`：文献结构化存储
- `digest.md`：可直接发到飞书/邮件/Notion 的摘要稿

## 偏好配置

编辑 `preferences.json`：

```json
{
  "likes": [
    {
      "weight": 2.5,
      "keywords": ["trapped ion", "quantum error correction"],
      "authors": ["Rainer Blatt"]
    }
  ]
}
```

- `weight`：你对这组偏好的重视程度。
- `keywords`：会在 title/summary 中匹配。
- `authors`：作者命中时额外加权。

## 重分类（定期梳理）

```bash
python -m src.lit_digest --recluster --db papers.db
```

适合每周/每月在分类规则调整后批量刷新。

## Zotero 集成

配置环境变量后，pipeline 会把 Top-N 高分文章同步到 Zotero：

```bash
export ZOTERO_API_KEY=xxxx
export ZOTERO_USER_ID=xxxx
export ZOTERO_COLLECTION_KEY=xxxx  # 可选
python -m src.lit_digest
```

## 下一步建议

- 用定时任务（crontab / GitHub Actions）每天跑一次。
- 增加向量召回（embedding）替换当前关键词匹配。
- 增加“已读反馈”回流，动态更新偏好画像。
- 加入 LLM 自动摘要与对比总结（按分类生成 weekly report）。

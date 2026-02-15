# trapped-ion

本项目提供一个**本地可运行**的文献更新与筛选系统：

- 更新推荐（抓取 + 打分 + 分类）
- 仅保留近期文献（默认 3 天，可配置）
- 已读/未读管理
- 按文章类型和主题标签筛选
- 勾选保存与导出到 Zotero
- 生成周报与总结

## 功能亮点

- **时间窗口保留**：默认只保留近 3 天文献，更新时会自动加入新文献并清理更早内容。
- **阅读状态**：每篇文献可标记为 `unread/read`。
- **文章类型识别**：自动打 `review/article/news` 标签。
- **小类标签识别**：如 `quantum-computing`、`quantum-simulation`、`theory`、`experiment` 等（可扩展）。
- **标签筛选阅读**：页面可按标签 + 已读状态过滤。

## 安装
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
cp subscriptions.example.json subscriptions.json
```

## 本地一键更新（默认保留3天）

```bash
./scripts/run_local_update.sh
```

自定义保留天数（例如 7 天）：

```bash
KEEP_DAYS=7 ./scripts/run_local_update.sh
```

## 本地页面（可视化筛选）

启动：

```bash
./scripts/run_dashboard.sh
```

打开 `http://127.0.0.1:8787`，可进行：

- 点击“更新推荐”（可设置保留天数）
- 按分类查看（分类内按相关度排序）
- 通过**标签**和**已读/未读**筛选
- 勾选并执行：保存、标记已读、标记未读、导出到 Zotero
- 一键生成周报

## Zotero 导出配置
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
```

## 命令行补充

```bash
# 主流程：只保留近3天
python -m src.lit_digest --history preferences.json --subscriptions subscriptions.json --keep-days 3

# 标记已读
python -m src.lit_digest --mark-read "arxiv||http://arxiv.org/abs/xxxx"

# 标记未读
python -m src.lit_digest --mark-unread "arxiv||http://arxiv.org/abs/xxxx"

# 生成周报
python -m src.lit_digest --weekly-report --db papers.db --report-output weekly_report.md --report-days 7
```
python -m src.lit_digest
```

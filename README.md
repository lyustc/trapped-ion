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

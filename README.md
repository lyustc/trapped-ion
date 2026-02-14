# trapped-ion

本项目提供一个**本地可运行**的文献更新与筛选系统：

- 更新推荐（抓取 + 打分 + 分类）
- 导出到 Zotero
- 生成周报与总结
- 本地可视化页面（更新按钮、分类/相关度排序、勾选保存）

> 按你的要求：不再依赖 GitHub 每日任务，主流程以本地脚本和本地页面为主。

## 功能

- 订阅源：arXiv + Nature 系列 + Science/Science Advances + PR 系列（PRL/PR Applied/PRX/PRX Quantum）。
- 偏好建模：关键词、作者、向量召回（embedding）混合打分。
- 存储：SQLite（`papers.db`）。
- 导出：可将筛选出的文献推送至 Zotero。
- 报告：生成 `digest.md` 和 `weekly_report.md`，可带 LLM 总结。
- 反馈回流：`like/dislike/save` 反馈可用于更新偏好。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp preferences.example.json preferences.json
cp subscriptions.example.json subscriptions.json
```

## 本地一键更新（推荐）

```bash
./scripts/run_local_update.sh
```

执行后更新：

- `papers.db`
- `digest.md`
- `weekly_report.md`

## 本地页面（更直观）

启动页面：

```bash
./scripts/run_dashboard.sh
```

打开：`http://127.0.0.1:8787`

页面提供：

- **更新推荐**按钮（抓取+打分+分类）
- 按**分类**展示，并在分类内按**相关度降序**
- 勾选文献并“**保存勾选（本地收藏）**”
- 一键“**导出已勾选到 Zotero**”
- 一键“**生成周报**”

## Zotero 导出配置

设置环境变量：

```bash
export ZOTERO_API_KEY=xxxx
export ZOTERO_USER_ID=xxxx
export ZOTERO_COLLECTION_KEY=xxxx  # 可选
```

## 周报与总结

命令行生成：

```bash
python -m src.lit_digest --weekly-report --db papers.db --report-output weekly_report.md --report-days 7
```

- 配置 `OPENAI_API_KEY` 时：自动附加 LLM 中文总结（按分类进展 + 对比 + 下周建议）
- 未配置时：输出结构化周报草稿

## 反馈回流（可选）

```bash
# 记录反馈
python -m src.lit_digest --feedback --db papers.db \
  --feedback-source arxiv --feedback-source-id http://arxiv.org/abs/xxxx \
  --feedback-action like --feedback-note "important"

# 应用反馈到偏好画像
python -m src.lit_digest --apply-feedback --db papers.db --history preferences.json
```

## 订阅范围修改

直接编辑 `subscriptions.json`，例如改 arXiv query 或增删 RSS 源。

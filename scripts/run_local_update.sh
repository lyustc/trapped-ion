#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m src.lit_digest --history preferences.json --subscriptions subscriptions.json
python -m src.lit_digest --weekly-report --db papers.db --report-output weekly_report.md

echo "更新完成：digest.md / weekly_report.md / papers.db"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PORT="${PORT:-8787}"
python -m src.web_app

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUNDLED_PYTHON="/Users/xuan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"

if [[ -x "$BUNDLED_PYTHON" ]]; then
  PYTHON="$BUNDLED_PYTHON"
else
  PYTHON="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON:-}" ]]; then
  echo "No usable Python found. Please run with the bundled Codex Python runtime." >&2
  exit 1
fi

"$PYTHON" "$PROJECT_ROOT/poc_cgm_remission/src/run_poc.py"
"$PYTHON" "$PROJECT_ROOT/poc_cgm_remission/src/build_dashboard.py"

echo
echo "Demo outputs:"
echo "  $PROJECT_ROOT/index.html"
echo "  $PROJECT_ROOT/poc_cgm_remission/reports/poc_summary.md"
echo "  $PROJECT_ROOT/poc_cgm_remission/reports/model_metrics.csv"
echo "  $PROJECT_ROOT/poc_cgm_remission/reports/data_quality_report.csv"
echo "  $PROJECT_ROOT/poc_cgm_remission/reports/audit_manifest.json"

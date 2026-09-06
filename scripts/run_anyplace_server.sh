#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
cd "$project_dir"
exec "$project_dir/_envs/anyplace/bin/python" scripts/serve_anyplace.py \
  --port "${ANYPLACE_PORT:-5590}" --record-dir "$project_dir/output/anyplace-rpc" "$@"

#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${ISAAC_SIM_PATH:-/root/isaacsim}/kit/python/lib:${LD_LIBRARY_PATH:-}"
export BUSAGENT_FLORENCE_MODEL="${BUSAGENT_FLORENCE_MODEL:-$(ls -d "$project_dir"/_models/florence-cache/snapshots/* | head -1)}"
export BUSAGENT_YOLOE_WEIGHTS="${BUSAGENT_YOLOE_WEIGHTS:-$project_dir/_models/yoloe-26x-seg.pt}"
cd "$project_dir"
test -f "$project_dir/_models/mobileclip2_b.ts"
test -f "$project_dir/_models/sam3.pt"
ln -sfn _models/mobileclip2_b.ts "$project_dir/mobileclip2_b.ts"
exec "$project_dir/_envs/vision/bin/python" scripts/serve_arena_vision.py "$@"

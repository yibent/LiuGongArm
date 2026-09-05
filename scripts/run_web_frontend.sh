#!/usr/bin/env bash
# Serve the prebuilt Web UI for GPU integration tests; never run source/HMR.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
frontend_dir="$project_dir/BusAgent/frontend"
node_bin="${BUSAGENT_NODE_BIN:-node}"
export BUSAGENT_PROXY_TARGET="${BUSAGENT_PROXY_TARGET:-http://127.0.0.1:3100}"

if [[ ! -f "$frontend_dir/dist/index.html" ]]; then
  echo "Missing frontend build: run pnpm build in BusAgent/frontend and deploy dist/." >&2
  exit 1
fi
if [[ ! -f "$frontend_dir/node_modules/vite/bin/vite.js" ]]; then
  echo "Missing Vite preview runtime in BusAgent/frontend/node_modules." >&2
  exit 1
fi

cd "$frontend_dir"
exec "$node_bin" "$frontend_dir/node_modules/vite/bin/vite.js" preview \
  --host "${BUSAGENT_WEB_HOST:-127.0.0.1}" \
  --port "${BUSAGENT_WEB_PORT:-5173}" --strictPort

#!/usr/bin/env bash
# GPUFree desktop launcher; uses Isaac's libpython before system libraries.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_dir="${ISAAC_SIM_PATH:-/root/isaacsim}"
export DISPLAY="${DISPLAY:-:20}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-xdg}"
export OMNI_KIT_ALLOW_ROOT=1
export LD_LIBRARY_PATH="${isaac_dir}/kit/python/lib:/opt/vulkan/x86_64/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
export HF_HOME="${project_dir}/.cache/huggingface"
export PYTHONUNBUFFERED=1
cd "$project_dir"
exec "$isaac_dir/python.sh" scripts/run_vision_follow.py --no-follow --control-host 127.0.0.1 --control-port 7861 "$@"

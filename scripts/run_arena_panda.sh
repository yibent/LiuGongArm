#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_dir="${ISAAC_SIM_PATH:-/root/isaacsim}"
unset PYTHONHOME PYTHONPATH
export LD_LIBRARY_PATH="${isaac_dir}/kit/python/lib:${isaac_dir}/kit/python/lib/python3.12/site-packages/torch/lib:/opt/vulkan/x86_64/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
export PYTHONEXE="${ARENA_PYTHON:-$project_dir/_envs/arena/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 PYTHONUNBUFFERED=1
export DISPLAY="${DISPLAY:-:20}" XAUTHORITY="${XAUTHORITY:-/home/ubuntu/.Xauthority}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-xdg}"
cd "$project_dir"
exec "$isaac_dir/python.sh" scripts/run_arena_panda.py "$@"

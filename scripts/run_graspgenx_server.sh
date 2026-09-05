#!/usr/bin/env bash
# Run in a separate Python environment; never install model packages into Isaac.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="$project_dir/_vendor/GraspGenX"
export GRASPGENX_CHECKPOINT_DIR="$project_dir/_models/graspgenx/checkpoints"
export GRASPGENX_GRIPPER_CFG_DIR="$model_dir/ext/gripper_descriptions"
export PYTHONUNBUFFERED=1
# NVRTC dynamically loads its builtins by name; expose this environment's
# matching CUDA runtime, not a system/Isaac CUDA version inherited over SSH.
model_site="$project_dir/_envs/graspgenx/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$model_site/nvidia/cuda_nvrtc/lib:$model_site/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$model_dir"
exec "$project_dir/_envs/graspgenx/bin/python" "$project_dir/scripts/serve_graspgenx.py" \
  --config "$GRASPGENX_CHECKPOINT_DIR/release" --assets_dir "$model_dir/assets" \
  --host 127.0.0.1 --port "${GRASPGENX_PORT:-5556}"

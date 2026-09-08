#!/usr/bin/env bash
set -euo pipefail
export DISPLAY="${DISPLAY:-:20}"
export XAUTHORITY="${XAUTHORITY:-/home/ubuntu/.Xauthority}"
# Both the desktop session and Supervisor may start us; one window per boot.
exec 9>/tmp/liugong-desktop-browser.lock
flock -n 9 || exit 0
while ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; do sleep 2; done
while ! curl --fail --silent http://127.0.0.1:7861/health >/dev/null; do sleep 2; done
while ! curl --fail --silent http://127.0.0.1:3100/v1/tasks/status >/dev/null; do sleep 2; done
exec /usr/bin/google-chrome --no-sandbox --disable-dev-shm-usage --no-first-run \
  --no-default-browser-check --user-data-dir=/root/.config/liugong-browser \
  --new-window 'http://127.0.0.1:8991/?workspace=1'

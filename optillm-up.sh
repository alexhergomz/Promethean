#!/usr/bin/env bash
# Start (or stop) the local optillm proxy in front of MiniMax /v1.
# Usage:  ./optillm-up.sh [start|stop|status|tail]
# Then in promethean:
#   /config minimax_base_url=http://127.0.0.1:8765/v1
#   /optillm cot_reflection       # or re2
set -euo pipefail

PORT=8765
LOG=~/.cache/promethean/optillm.log
PID=~/.cache/promethean/optillm.pid
UPSTREAM=https://api.minimax.io/v1

case "${1:-status}" in
  start)
    mkdir -p "$(dirname "$LOG")"
    if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
      echo "already running (pid $(cat "$PID")) on :$PORT"
      exit 0
    fi
    if [[ -z "${MINIMAX_API_KEY:-}" ]]; then
      echo "error: MINIMAX_API_KEY not set" >&2; exit 1
    fi
    # Apply local optillm patches idempotently (M2-compat fixes).
    # See patches/apply_optillm_patches.py.
    HERE="$(cd "$(dirname "$0")" && pwd)"
    if [[ -f "$HERE/patches/apply_optillm_patches.py" ]]; then
      ~/miniconda3/bin/python "$HERE/patches/apply_optillm_patches.py" || \
        echo "warning: patch apply step had issues — continuing" >&2
    fi
    # optillm forwards bearer auth via OPENAI_API_KEY env var.
    OPENAI_API_KEY="$MINIMAX_API_KEY" nohup \
      ~/miniconda3/bin/optillm --base-url "$UPSTREAM" --port "$PORT" --log info \
      >"$LOG" 2>&1 &
    echo $! >"$PID"
    echo "starting optillm on :$PORT (pid $(cat "$PID"), log $LOG)"
    echo "give it ~15s to load plugins before sending requests"
    ;;
  stop)
    if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
      kill "$(cat "$PID")" && rm -f "$PID"
      echo "stopped"
    else
      echo "(not running)"
    fi
    ;;
  status)
    if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
      echo "running pid $(cat "$PID") on :$PORT"
      ss -tln 2>/dev/null | grep ":$PORT " || true
    else
      echo "not running"
    fi
    ;;
  tail)
    tail -f "$LOG"
    ;;
  *)
    echo "usage: $0 {start|stop|status|tail}" >&2; exit 2
    ;;
esac

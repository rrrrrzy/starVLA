#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
STORE=/inspire/qb-ilm2/project/26summer-camp-10/public/ten/
LOG_BASE="$STORE/log"

if [ ! -d "$LOG_BASE" ]; then
    echo "[INFO] No log directory found: $LOG_BASE"
    exit 0
fi

LATEST=$(ls -d "$LOG_BASE"/[0-9]* 2>/dev/null | sort | tail -1)

if [ -z "$LATEST" ]; then
    echo "[INFO] No timestamp directories found under $LOG_BASE"
    exit 0
fi

echo "[INFO] Stopping processes for: $(basename "$LATEST")"

KILLED=0
SKIPPED=0

for pid_file in $(find "$LATEST" -name '*.pid' | sort); do
    name=$(basename "$pid_file" .pid)
    if [ ! -f "$pid_file" ]; then
        continue
    fi
    PID=$(cat "$pid_file")
    if [ -z "$PID" ]; then
        rm -f "$pid_file"
        continue
    fi
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[KILL] $name (PID=$PID)"
        pkill -P "$PID" 2>/dev/null || true
        kill "$PID" 2>/dev/null || true
        KILLED=$((KILLED + 1))
    else
        echo "[SKIP] $name (PID=$PID, already dead)"
        SKIPPED=$((SKIPPED + 1))
    fi
    rm -f "$pid_file"
done

echo
echo "[DONE] Killed=$KILLED, Skipped=$SKIPPED"
echo "[INFO] PID files cleaned: $LATEST"

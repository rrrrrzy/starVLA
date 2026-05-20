#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
LOG_BASE="$REPO/log"
TARGET_DIR="/inspire/qb-ilm2/project/26summer-camp-10/public/ten/log/"

if [ ! -d "$LOG_BASE" ]; then
    echo "[INFO] No log directory found: $LOG_BASE"
    exit 0
fi

LATEST=$(ls -d "$LOG_BASE"/[0-9]* 2>/dev/null | sort | tail -1)

if [ -z "$LATEST" ]; then
    echo "[INFO] No timestamp directories found under $LOG_BASE"
    exit 0
fi

echo "[INFO] Copying vedios for: $(basename "$LATEST")"

cp -r "$LATEST" "$TARGET_DIR/"

echo
echo "[DONE] Copied=$LATEST to DIR=$TARGET_DIR"

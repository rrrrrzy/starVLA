#!/usr/bin/env bash
set -euo pipefail

BASE=/inspire/qb-ilm2/project/26summer-camp-10/26220056
REPO=$BASE/starVLA

TIMESTAMP=$(date +%Y%m%d%H%M)
LOG_BASE=$REPO/log/$TIMESTAMP
LOG_DIR=$LOG_BASE/starVLA

CKPT=/inspire/qb-ilm2/project/26summer-camp-10/public/ten/ckpt/v0519/qwen35_2b_pi_calvin_abc_multiview_20260518_174511/checkpoints/steps_20000_pytorch_model.pt

# 使用哪些 GPU
GPUS=(0 1 2 3 4 5 6 7)

# 起始端口：GPU0 -> 5694, GPU1 -> 5695, ...
BASE_PORT=5694

mkdir -p "$LOG_DIR"

for idx in "${!GPUS[@]}"; do
    GPU="${GPUS[$idx]}"
    PORT=$((BASE_PORT + idx))

    LOG_FILE="$LOG_DIR/starvla_server_gpu${GPU}_port${PORT}.log"
    PID_FILE="$LOG_DIR/starvla_server_gpu${GPU}_port${PORT}.pid"

    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "[SKIP] GPU=$GPU PORT=$PORT already running, PID=$OLD_PID"
            echo "       Log: $LOG_FILE"
            continue
        fi
    fi

    echo "[START] GPU=$GPU PORT=$PORT"

    nohup bash -lc "
source $BASE/.venvs/starVLA/bin/activate
cd \$REPO

CUDA_VISIBLE_DEVICES=$GPU python -u deployment/model_server/server_policy.py \
    --ckpt_path $CKPT \
    --port $PORT \
    --use_bf16
" > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > "$PID_FILE"

    echo "[OK] GPU=$GPU PORT=$PORT PID=$PID"
    echo "     Log: $LOG_FILE"
done

echo
echo "查看所有 server："
echo "ps aux | grep server_policy.py | grep -v grep"
echo
echo "日志目录: $LOG_DIR"
echo "查看某个日志，例如："
echo "tail -f $LOG_DIR/starvla_server_gpu${GPUS[0]}_port${BASE_PORT}.log"
#!/usr/bin/env bash
set -euo pipefail

BASE=/inspire/qb-ilm2/project/26summer-camp-10/26220056
REPO=$BASE/starVLA

TIMESTAMP=$(date +%Y%m%d%H%M)
LOG_BASE=$REPO/log/$TIMESTAMP
LOG_DIR=$LOG_BASE/calvin
RUN_DIR=$BASE/runs/calvin_parallel
SPLIT_DIR=$RUN_DIR/eval_splits

CKPT=/inspire/qb-ilm2/project/26summer-camp-10/public/ten/ckpt/v0519/qwen35_2b_gr00t_calvin_abc_multiview_d_style_ft_20260519_084009/checkpoints/steps_2000_pytorch_model.pt
DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/26220056/calvin/dataset/calvin_debug_dataset
CALVIN_CONFIG_PATH=/inspire/qb-ilm2/project/26summer-camp-10/26220056/calvin/calvin_models/conf
SOURCE_EVAL_SEQUENCES=$REPO/examples/calvin/eval_files/eval_sequences.json

HOST=127.0.0.1
BASE_PORT=5694
UNNORM_KEY=franka

# 使用哪些 GPU / worker
GPUS=(0 1 2 3 4 5 6 7)
NUM_WORKERS=${#GPUS[@]}

# 调试时可以设成 10；正式跑全量就改成空：
# LIMIT_SEQUENCES=""
LIMIT_SEQUENCES=1000

mkdir -p "$LOG_DIR" "$RUN_DIR" "$SPLIT_DIR"

echo "[INFO] Splitting eval sequences..."

if [ -n "${LIMIT_SEQUENCES}" ]; then
    python "$BASE/starVLA/scripts/split_calvin_sequences.py" \
        --input "$SOURCE_EVAL_SEQUENCES" \
        --out-dir "$SPLIT_DIR" \
        --num-workers "$NUM_WORKERS" \
        --limit "$LIMIT_SEQUENCES"
else
    python "$BASE/starVLA/scripts/split_calvin_sequences.py" \
        --input "$SOURCE_EVAL_SEQUENCES" \
        --out-dir "$SPLIT_DIR" \
        --num-workers "$NUM_WORKERS"
fi

echo
echo "[INFO] Starting Calvin eval workers..."

for idx in "${!GPUS[@]}"; do
    GPU="${GPUS[$idx]}"
    PORT=$((BASE_PORT + idx))

    EVAL_SEQ="$SPLIT_DIR/eval_sequences_worker_${idx}.json"
    COUNT_FILE="$SPLIT_DIR/eval_sequences_worker_${idx}.count"

    if [ ! -f "$EVAL_SEQ" ]; then
        echo "[WARN] Missing split file: $EVAL_SEQ, skip worker $idx"
        continue
    fi

    NUM_SEQUENCES=$(cat "$COUNT_FILE")

    if [ "$NUM_SEQUENCES" -le 0 ]; then
        echo "[SKIP] worker=$idx has 0 sequences"
        continue
    fi

    WORK_DIR="$RUN_DIR/worker_${idx}"
    mkdir -p "$WORK_DIR"

    LOG_FILE="$LOG_DIR/calvin_eval_worker${idx}_gpu${GPU}_port${PORT}.log"
    PID_FILE="$LOG_DIR/calvin_eval_worker${idx}_gpu${GPU}_port${PORT}.pid"

    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "[SKIP] worker=$idx already running, PID=$OLD_PID"
            echo "       Log: $LOG_FILE"
            continue
        fi
    fi

    echo "[START] worker=$idx GPU=$GPU PORT=$PORT NUM_SEQUENCES=$NUM_SEQUENCES"

    setsid bash -lc "
source $BASE/env_calvin.sh

# 每个 worker 用独立工作目录，避免 tmp/calvin/eval_logs 互相覆盖
cd $WORK_DIR

# 让 Python 能找到 StarVLA / CALVIN 源码
export PYTHONPATH=$REPO:$BASE/calvin/calvin_models:$BASE/calvin/calvin_env:$BASE/calvin/calvin_env/tacto:\${PYTHONPATH:-}

# CALVIN / PyBullet EGL rendering
unset DISPLAY
export PYOPENGL_PLATFORM=egl
export EGL_VISIBLE_DEVICES=$GPU
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Video recording (set CALVIN_SAVE_VIDEO=1 to save all videos)
export CALVIN_SAVE_VIDEO=${CALVIN_SAVE_VIDEO:-1}
export CALVIN_VIDEO_CAMERA=${CALVIN_VIDEO_CAMERA:-rgb_static}
export CALVIN_VIDEO_DIR=$LOG_DIR/videos

python -u $REPO/examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained_path $CKPT \
    --args.unnorm_key $UNNORM_KEY \
    --args.host $HOST \
    --args.port $PORT \
    --args.dataset_path $DATASET_PATH \
    --args.calvin_config_path $CALVIN_CONFIG_PATH \
    --args.eval_sequences_path $EVAL_SEQ \
    --args.num_sequences $NUM_SEQUENCES
" > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > "$PID_FILE"

    echo "[OK] worker=$idx PID=$PID"
    echo "     Log: $LOG_FILE"
done

echo
echo "查看所有 eval："
echo "ps aux | grep eval_calvin.py | grep -v grep"
echo
echo "日志目录: $LOG_DIR"
echo "查看日志，例如："
echo "tail -f $LOG_DIR/calvin_eval_worker0_gpu${GPUS[0]}_port${BASE_PORT}.log"
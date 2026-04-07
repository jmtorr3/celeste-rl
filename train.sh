#!/usr/bin/env bash
# Train a Celeste RL agent
# Usage: ./train.sh [options]
#   -v VERSION    Version to train: 1, 2, 3, ... (default: 1)
#   -e EPISODES   Number of episodes (default: 3000)
#   -s MAX_STEPS  Max steps per episode (default: 500)  [v1 only]
#   -r ROOM       Room number 0-30 (default: 0)         [v1 only]
#   -l LR         Learning rate (default: 5e-4)         [v1 only]
#   -m MODEL      Load existing model to resume training [v1 only]
#   --eval-only   Only run evaluation, skip training    [v1 only]

set -e
cd "$(dirname "$0")"

VERSION=1
EPISODES=3000
MAX_STEPS=500
ROOM=0
LR=5e-4
MODEL=""
EVAL_ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v)  VERSION="$2"; shift 2 ;;
        -e)  EPISODES="$2"; shift 2 ;;
        -s)  MAX_STEPS="$2"; shift 2 ;;
        -r)  ROOM="$2"; shift 2 ;;
        -l)  LR="$2"; shift 2 ;;
        -m)  MODEL="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY="--eval-only"; shift ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# //'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

source venv/bin/activate 2>/dev/null || true

# Route to the correct script based on version
case "$VERSION" in
    1)
        SCRIPT="src/train.py"
        ARGS="--episodes $EPISODES --max-steps $MAX_STEPS --room $ROOM --lr $LR"
        [[ -n "$MODEL" ]] && ARGS="$ARGS --model $MODEL"
        [[ -n "$EVAL_ONLY" ]] && ARGS="$ARGS --eval-only"
        ;;
    2)
        SCRIPT="src/train_v2.py"
        ARGS="--episodes $EPISODES"
        ;;
    *)
        SCRIPT="src/train_v${VERSION}.py"
        if [[ ! -f "$SCRIPT" ]]; then
            echo "Error: $SCRIPT not found. Create it first (see WORKFLOW.md)."
            exit 1
        fi
        ARGS="--episodes $EPISODES"
        ;;
esac

echo "==> Training v${VERSION}: python $SCRIPT $ARGS"
python "$SCRIPT" $ARGS

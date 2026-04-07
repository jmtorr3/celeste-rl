#!/usr/bin/env bash
# Train a Celeste RL agent
# Usage: ./train.sh [options]
#   -v2               Use train_v2 (exploration bonuses)
#   -e EPISODES       Number of episodes (default: 3000)
#   -s MAX_STEPS      Max steps per episode (default: 500)
#   -r ROOM           Room number 0-30 (default: 0)
#   -l LR             Learning rate (default: 5e-4)
#   -m MODEL          Load existing model to continue training
#   --eval-only       Only run evaluation

set -e
cd "$(dirname "$0")"

VERSION=""
EPISODES=3000
MAX_STEPS=500
ROOM=0
LR=5e-4
MODEL=""
EVAL_ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v2) VERSION="v2"; shift ;;
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

if [[ "$VERSION" == "v2" ]]; then
    SCRIPT="src/train_v2.py"
    ARGS="--episodes $EPISODES"
    echo "Training V2 (exploration bonuses) with: python $SCRIPT $ARGS"
else
    SCRIPT="src/train.py"
    ARGS="--episodes $EPISODES --max-steps $MAX_STEPS --room $ROOM --lr $LR"
    [[ -n "$MODEL" ]] && ARGS="$ARGS --model $MODEL"
    [[ -n "$EVAL_ONLY" ]] && ARGS="$ARGS --eval-only"
    echo "Training with: python $SCRIPT $ARGS"
fi

python "$SCRIPT" $ARGS

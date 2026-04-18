#!/usr/bin/env bash
# Watch a trained Celeste RL agent play
# Usage: ./watch.sh [options]
#   -v VERSION    Version to watch: 1, 2, 3, ... (default: 1)
#   -m MODEL      Override model path directly
#   -r ROOM       Room number 0-30 (default: 0)
#   -e EPISODES   Number of episodes to watch (default: 3)
#   -d DELAY      Seconds between frames (default: 0.03)

set -e
cd "$(dirname "$0")"

VERSION=1
MODEL=""
ROOM=0
EPISODES=3
DELAY=0.03

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v) VERSION="$2"; shift 2 ;;
        -m) MODEL="$2"; shift 2 ;;
        -r) ROOM="$2"; shift 2 ;;
        -e) EPISODES="$2"; shift 2 ;;
        -d) DELAY="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,8p' "$0" | sed 's/^# //'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Resolve model path from version if not given explicitly
if [[ -z "$MODEL" ]]; then
    case "$VERSION" in
        1) MODEL="models/dqn_best.pt" ;;
        2) MODEL="models/model_v2_best.pt" ;;
        *) MODEL="models/v${VERSION}_best.pt" ;;
    esac
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: model not found at $MODEL"
    echo "Train it first with: ./train.sh -v $VERSION"
    exit 1
fi

# v3+ models use DuelingDQN — pass --dueling automatically
DUELING=""
if [[ "$VERSION" -ge 3 ]] || [[ "$MODEL" == *"v3"* ]]; then
    DUELING="--dueling"
fi

source venv/bin/activate 2>/dev/null || true

echo "==> Watching v${VERSION}: $MODEL  room=$ROOM  episodes=$EPISODES  delay=${DELAY}s"
python scripts/watch_agent.py --model "$MODEL" --room "$ROOM" --episodes "$EPISODES" --delay "$DELAY" $DUELING

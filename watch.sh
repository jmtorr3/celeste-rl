#!/usr/bin/env bash
# Watch a trained Celeste RL agent play
# Usage: ./watch.sh [options]
#   -v VERSION    Version to watch: 1, 2, 3, ... (default: 1)
#   -i RUN_ID     Run ID — loads runs/{RUN_ID}/best.pt
#   -m MODEL      Override model path directly
#   -r ROOM       Room number 0-30 (default: 0)
#   -e EPISODES   Number of episodes to watch (default: 3)
#   -d DELAY      Seconds between frames (default: 0.03)

set -e
cd "$(dirname "$0")"

VERSION=1
MODEL=""
RUN_ID_OVERRIDE=""
ROOM=0
EPISODES=3
DELAY=0.03

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v) VERSION="$2"; shift 2 ;;
        -i) RUN_ID_OVERRIDE="$2"; shift 2 ;;
        -m) MODEL="$2"; shift 2 ;;
        -r) ROOM="$2"; shift 2 ;;
        -e) EPISODES="$2"; shift 2 ;;
        -d) DELAY="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,9p' "$0" | sed 's/^# //'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# -i RUN_ID overrides the version-based resolution
if [[ -n "$RUN_ID_OVERRIDE" && -z "$MODEL" ]]; then
    MODEL="runs/${RUN_ID_OVERRIDE}/best.pt"
fi

# Resolve model path from version if not given explicitly.
# Prefer new layout runs/{run_id}/best.pt; fall back to legacy models/* paths.
if [[ -z "$MODEL" ]]; then
    case "$VERSION" in
        1) RUN_ID="dqn"; LEGACY="models/dqn_best.pt" ;;
        2) RUN_ID="v2"; LEGACY="models/model_v2_best.pt" ;;
        *) RUN_ID="v${VERSION}"; LEGACY="models/v${VERSION}_best.pt" ;;
    esac
    if [[ -f "runs/${RUN_ID}/best.pt" ]]; then
        MODEL="runs/${RUN_ID}/best.pt"
    else
        MODEL="$LEGACY"
    fi
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: model not found at $MODEL"
    echo "Train it first with: ./train.sh -v $VERSION"
    exit 1
fi

# v3 / curriculum models use DuelingDQN — pass --dueling automatically.
# (Hybrid and BC switched to plain DQN as of late 2026-04 to match dqn_r1.
# Old hybrid/bc checkpoints still need --dueling, but new ones don't, so we
# leave them off the auto-list — pass it explicitly if needed.)
DUELING=""
if [[ "$VERSION" -ge 3 ]] || [[ "$MODEL" == *"v3"* ]] || [[ "$MODEL" == *"curriculum"* ]]; then
    DUELING="--dueling"
fi

source venv/bin/activate 2>/dev/null || true

echo "==> Watching v${VERSION}: $MODEL  room=$ROOM  episodes=$EPISODES  delay=${DELAY}s"
python scripts/watch_agent.py --model "$MODEL" --room "$ROOM" --episodes "$EPISODES" --delay "$DELAY" $DUELING

#!/usr/bin/env bash
# Watch a trained Celeste RL agent play
# Usage: ./watch.sh [options]
#   -m MODEL      Model file to load (default: models/dqn_best.pt)
#   -r ROOM       Room number 0-30 (default: 0)
#   -e EPISODES   Number of episodes to watch (default: 3)
#   -d DELAY      Seconds between frames (default: 0.03)

set -e
cd "$(dirname "$0")"

MODEL="models/dqn_best.pt"
ROOM=0
EPISODES=3
DELAY=0.03

while [[ $# -gt 0 ]]; do
    case "$1" in
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

source venv/bin/activate 2>/dev/null || true

echo "Watching: $MODEL  room=$ROOM  episodes=$EPISODES  delay=${DELAY}s"
python scripts/watch_agent.py --model "$MODEL" --room "$ROOM" --episodes "$EPISODES" --delay "$DELAY"

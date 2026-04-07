#!/usr/bin/env bash
# Sync models/ to/from Google Drive using rclone
# Usage: ./sync_models.sh push|pull|status
#
# Setup (once):
#   brew install rclone
#   rclone config   → follow prompts, name the remote "gdrive"
#   Then set REMOTE below to your Drive folder path.

set -e
cd "$(dirname "$0")"

REMOTE="gdrive:celeste-rl/models"
LOCAL="models/"

CMD="${1:-}"

case "$CMD" in
    push)
        echo "==> Uploading models/ → $REMOTE"
        rclone copy "$LOCAL" "$REMOTE" --progress --exclude ".gitkeep"
        echo "Done."
        ;;
    pull)
        echo "==> Downloading $REMOTE → models/"
        rclone copy "$REMOTE" "$LOCAL" --progress
        echo "Done."
        ;;
    status)
        echo "==> Comparing local models/ with $REMOTE"
        rclone check "$LOCAL" "$REMOTE" --one-way 2>&1 | grep -E "ERROR|INFO|Differences" || true
        echo ""
        echo "Local files:"
        ls -lh "$LOCAL"*.pt 2>/dev/null || echo "  (none)"
        echo ""
        echo "Remote files:"
        rclone ls "$REMOTE" 2>/dev/null || echo "  (none or rclone not configured)"
        ;;
    *)
        echo "Usage: ./sync_models.sh push|pull|status"
        echo ""
        echo "  push    Upload local models/ to Google Drive"
        echo "  pull    Download models from Google Drive"
        echo "  status  Show diff between local and remote"
        echo ""
        echo "First-time setup:"
        echo "  brew install rclone"
        echo "  rclone config   # create a remote named 'gdrive'"
        echo "  Edit REMOTE= in this script to set your Drive folder path"
        exit 1
        ;;
esac

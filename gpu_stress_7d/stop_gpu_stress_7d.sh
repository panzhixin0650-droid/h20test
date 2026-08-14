#!/usr/bin/env bash
set -Eeuo pipefail

session_name="${1:-${SESSION_NAME:-gpu-stress-7d}}"
if ! tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session '$session_name' does not exist" >&2
    exit 1
fi

tmux send-keys -t "$session_name:0.0" C-c
echo "Sent a graceful stop request to tmux session '$session_name'."
echo "The session will close after the current CUDA operation exits."

#!/bin/bash

# -----------------------------------------------------------------------------
# SSH Tunnel Helper
# -----------------------------------------------------------------------------
# Purpose:
#   Create a secure SSH tunnel to a remote server running the live trading
#   dashboard (FastAPI on port 8000). This lets you access the dashboard at
#   http://localhost:<LOCAL_PORT> on your laptop while the server runs 24/7.
#
# Usage:
#   1) Make executable (first time only):
#        chmod +x ssh_tunnel.sh
#
#   2) Start tunnel (recommended - pass host/user):
#        ./ssh_tunnel.sh start --host <SERVER_IP> --user <SSH_USER> \
#           [--local 8000] [--remote 8000]
#      Then open: http://localhost:8000
#
#   3) Check status:
#        ./ssh_tunnel.sh status
#
#   4) Stop tunnel:
#        ./ssh_tunnel.sh stop
#
#   5) Optional: environment variables instead of flags
#        export SSH_TUNNEL_HOST=<SERVER_IP>
#        export SSH_TUNNEL_USER=<SSH_USER>
#        export SSH_TUNNEL_LOCAL=8000
#        export SSH_TUNNEL_REMOTE=8000
#        ./ssh_tunnel.sh start
#
# Notes:
#   - The tunnel is local: it forwards your laptop's localhost:<LOCAL_PORT>
#     to remote 127.0.0.1:<REMOTE_PORT> on the server via SSH.
#   - If autossh is installed, it's used automatically for auto-reconnect.
#   - The script stores its PID in /tmp/ssh_tunnel_<LOCAL_PORT>.pid and logs to
#     <project_root>/logs/ssh_tunnel_<LOCAL_PORT>.log
# -----------------------------------------------------------------------------

set -euo pipefail

# Defaults (can be overridden by flags or env)
HOST="${SSH_TUNNEL_HOST:-}"
USER_NAME="${SSH_TUNNEL_USER:-}"
LOCAL_PORT="${SSH_TUNNEL_LOCAL:-8000}"
REMOTE_PORT="${SSH_TUNNEL_REMOTE:-8000}"

# Resolve project root (this script sits in live_trading/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="/tmp/ssh_tunnel_${LOCAL_PORT}.pid"
LOG_FILE="$LOG_DIR/ssh_tunnel_${LOCAL_PORT}.log"

mkdir -p "$LOG_DIR"

print_usage() {
	echo "Usage: $0 {start|stop|status} [--host HOST] [--user USER] [--local LPORT] [--remote RPORT]"
}

parse_args() {
	CMD=""
	while [[ $# -gt 0 ]]; do
		case "$1" in
			start|stop|status)
				CMD="$1"; shift ;;
			--host)
				HOST="$2"; shift 2 ;;
			--user)
				USER_NAME="$2"; shift 2 ;;
			--local)
				LOCAL_PORT="$2"; PID_FILE="/tmp/ssh_tunnel_${LOCAL_PORT}.pid"; LOG_FILE="$LOG_DIR/ssh_tunnel_${LOCAL_PORT}.log"; shift 2 ;;
			--remote)
				REMOTE_PORT="$2"; shift 2 ;;
			-h|--help)
				print_usage; exit 0 ;;
			*)
				echo "Unknown arg: $1"; print_usage; exit 1 ;;
		esac
	done
	if [[ -z "$CMD" ]]; then
		print_usage; exit 1
	fi
	COMMAND="$CMD"
}

is_running() {
	if [[ -f "$PID_FILE" ]]; then
		PID="$(cat "$PID_FILE" 2>/dev/null || true)"
		if [[ -n "$PID" ]] && ps -p "$PID" >/dev/null 2>&1; then
			return 0
		fi
	fi
	return 1
}

start_tunnel() {
	if [[ -z "$HOST" || -z "$USER_NAME" ]]; then
		echo "Error: host/user not specified. Use --host/--user or SSH_TUNNEL_HOST/SSH_TUNNEL_USER env vars." >&2
		exit 1
	fi

	if is_running; then
		echo "Tunnel already running (PID $(cat "$PID_FILE"))."
		return 0
	fi

	# Prefer autossh if available for auto-reconnect
	if command -v autossh >/dev/null 2>&1; then
		SSH_CMD="autossh -M 0"
	else
		SSH_CMD="ssh"
	fi

	SSH_OPTS=(
		-o ServerAliveInterval=60
		-o ServerAliveCountMax=3
		-o ExitOnForwardFailure=yes
		-N
		-L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"
	)

	echo "Starting SSH tunnel: localhost:${LOCAL_PORT} -> ${HOST}:127.0.0.1:${REMOTE_PORT}"
	echo "Logging to: $LOG_FILE"

	# Run in background, capture PID
	# shellcheck disable=SC2086
	nohup $SSH_CMD ${SSH_OPTS[*]} "${USER_NAME}@${HOST}" >>"$LOG_FILE" 2>&1 &
	PID=$!
	echo "$PID" > "$PID_FILE"
	sleep 1
	if ps -p "$PID" >/dev/null 2>&1; then
		echo "Tunnel started (PID $PID). Open http://localhost:${LOCAL_PORT}"
	else
		echo "Failed to start tunnel. Check logs: $LOG_FILE" >&2
		exit 1
	fi
}

stop_tunnel() {
	if is_running; then
		PID="$(cat "$PID_FILE")"
		echo "Stopping tunnel (PID $PID)..."
		kill "$PID" 2>/dev/null || true
		sleep 1
		if ps -p "$PID" >/dev/null 2>&1; then
			echo "Force killing (PID $PID)..."
			kill -9 "$PID" 2>/dev/null || true
		fi
		rm -f "$PID_FILE"
		echo "Stopped."
	else
		echo "Tunnel not running."
	fi
}

status_tunnel() {
	if is_running; then
		echo "Tunnel running (PID $(cat "$PID_FILE")) -> http://localhost:${LOCAL_PORT}"
	else
		echo "Tunnel not running."
	fi
}

main() {
	parse_args "$@"
	case "$COMMAND" in
		start) start_tunnel ;;
		stop) status_tunnel; stop_tunnel ;;
		status) status_tunnel ;;
		*) print_usage; exit 1 ;;
	esac
}

main "$@"

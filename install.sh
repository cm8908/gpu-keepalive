#!/usr/bin/env bash
# Install gpu-keepalive as a systemd service.
#
#   sudo ./install.sh              system-wide service, runs as $SUDO_USER
#   ./install.sh --user            per-user service, no root needed
#
# A per-user service only runs while you are logged in unless lingering is enabled:
#   loginctl enable-linger "$USER"
set -euo pipefail

MODE=system
[[ "${1:-}" == "--user" ]] && MODE=user

SRC="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"

if [[ $MODE == system ]]; then
  [[ $EUID -eq 0 ]] || { echo "run as root, or use: ./install.sh --user" >&2; exit 1; }
  INSTALL_DIR="${INSTALL_DIR:-/opt/gpu-keepalive}"
  CONFIG_PATH="${CONFIG_PATH:-/etc/gpu-keepalive.toml}"
  STATE_DIR="${STATE_DIR:-/var/lib/gpu-keepalive}"
  BIN_DIR="${BIN_DIR:-/usr/local/bin}"
  UNIT_DIR=/etc/systemd/system
  RUN_USER="${RUN_USER:-${SUDO_USER:-root}}"
  IDENTITY="User=$RUN_USER"$'\n'"Group=$(id -gn "$RUN_USER")"
  TARGET=multi-user.target
  SYSTEMCTL=(systemctl)
else
  INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/share/gpu-keepalive}"
  CONFIG_PATH="${CONFIG_PATH:-$HOME/.config/gpu-keepalive.toml}"
  STATE_DIR="${STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/gpu-keepalive}"
  BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  IDENTITY="Environment=GPU_KEEPALIVE_STATE_DIR=$STATE_DIR"
  TARGET=default.target
  SYSTEMCTL=(systemctl --user)
fi

"$PYTHON" -c "import pynvml" 2>/dev/null \
  || { echo "missing NVML bindings: $PYTHON -m pip install nvidia-ml-py" >&2; exit 1; }
"$PYTHON" -c "import torch" 2>/dev/null \
  || echo "warning: torch not importable by $PYTHON -- the daemon will start but every worker will fail" >&2

echo "== installing to $INSTALL_DIR ($MODE mode)"
mkdir -p "$INSTALL_DIR" "$STATE_DIR/holds" "$BIN_DIR" "$UNIT_DIR"
rm -rf "$INSTALL_DIR/gpu_keepalive"
cp -r "$SRC/gpu_keepalive" "$SRC/bin" "$INSTALL_DIR/"
find "$INSTALL_DIR/gpu_keepalive" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
chmod +x "$INSTALL_DIR"/bin/*

if [[ -f "$CONFIG_PATH" ]]; then
  echo "== keeping existing config at $CONFIG_PATH"
else
  cp "$SRC/config/gpu-keepalive.toml" "$CONFIG_PATH"
  echo "== wrote config to $CONFIG_PATH"
fi

ln -sf "$INSTALL_DIR/bin/gpuidle" "$BIN_DIR/gpuidle"
ln -sf "$INSTALL_DIR/bin/gpu-run" "$BIN_DIR/gpu-run"

if [[ $MODE == system ]]; then
  chown -R "$RUN_USER" "$STATE_DIR"
fi

python3 - "$SRC/systemd/gpu-keepalive.service" "$UNIT_DIR/gpu-keepalive.service" \
         "$IDENTITY" "$INSTALL_DIR" "$PYTHON" "$TARGET" <<'PY'
import sys
src, dst, identity, root, python, target = sys.argv[1:7]
text = open(src).read()
for token, value in (("__IDENTITY__", identity), ("__ROOT__", root),
                     ("__PYTHON__", python), ("__TARGET__", target)):
    text = text.replace(token, value)
open(dst, "w").write(text)
PY

"${SYSTEMCTL[@]}" daemon-reload
"${SYSTEMCTL[@]}" enable gpu-keepalive.service >/dev/null
echo "== unit installed at $UNIT_DIR/gpu-keepalive.service"
echo
echo "start:   ${SYSTEMCTL[*]} start gpu-keepalive"
echo "status:  gpuidle status"
if [[ $MODE == user ]]; then
  echo "logs:    journalctl --user -u gpu-keepalive -f"
  echo
  echo "note: a per-user service stops when your last session ends."
  echo "      keep it running across logouts with: loginctl enable-linger \"$USER\""
else
  echo "logs:    journalctl -u gpu-keepalive -f"
fi

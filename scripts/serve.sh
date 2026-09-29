#!/usr/bin/env bash
# Start the OpenAI-compatible AiiStream server on 127.0.0.1 (port from server/config.json, default 8081).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -x "$ROOT/server/bin/thermal_state" ]; then
  echo "building thermal-state helper (one time)…"; "$ROOT/server/scripts/build_thermal_helper.sh"
fi
cd "$ROOT/server"
exec "${PYTHON:-python3}" -m aiistream_server --config "$ROOT/server/config.json"

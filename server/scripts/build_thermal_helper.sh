#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/bin"
swiftc "$ROOT/thermal_state.swift" -O -o "$ROOT/bin/thermal_state"
echo "built $ROOT/bin/thermal_state"

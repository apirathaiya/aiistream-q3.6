#!/usr/bin/env bash
# Download the exact checkpoint AiiStream was validated against, and verify every byte.
# Requires: python3 with huggingface_hub >= 1.0 (`pip install -U huggingface_hub`), ~20 GB free.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PINS="$ROOT/scripts/model_pins.json"
PY="${PYTHON:-python3}"
"$PY" - "$ROOT" "$PINS" <<'PYEOF'
import hashlib, json, os, shutil, sys
from huggingface_hub import hf_hub_download
root, pins = sys.argv[1], json.load(open(sys.argv[2]))
ckpt, meta = os.path.join(root, "model", "checkpoint"), os.path.join(root, "model", "hf-metadata")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def fetch(name, dest_dir, expected):
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest) and sha(dest) == expected:
        print(f"ok (cached)  {name}"); return
    p = hf_hub_download(pins["repo"], name, revision=pins["revision"])
    shutil.copyfile(p, dest)
    got = sha(dest)
    if got != expected: sys.exit(f"SHA-256 MISMATCH for {name}: {got} != {expected}")
    print(f"ok (verified) {name}")
for name, s in pins["shards"].items(): fetch(name, ckpt, s["sha256"])
for name, h in pins["metadata"].items(): fetch(name, meta, h)
print("checkpoint ready and byte-verified at revision", pins["revision"])
PYEOF

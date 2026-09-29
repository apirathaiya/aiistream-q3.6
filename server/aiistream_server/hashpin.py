from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_SHA256 = "2c18a53ba204e89e231aa755c8fe2ccf84d6631b1135341acd4e79217d467b2c"
PARALLEL_SHA256 = "cb97e9d1d1e78fe8b6a3d32d5d2f1d439e083cd8f13a995434aa2c969ba694d1"
DIRECT_SHA256 = "85617b0466213f37b9cad19aa906ef07ae67046c4281e30eaadff6a1f723ac41"
PREFETCH_SHA256 = "24b51de20e45ae16c11fe87ca23d1c5adc671175183a635659b2ebcc9af3a45c"
PROMOTED_RUNTIME_SHA256 = PARALLEL_SHA256

PINNED_RUNTIME_SHA256 = {
    "aiistream_model.py": MODEL_SHA256,
    "aiistream_parallel.py": PARALLEL_SHA256,
    "aiistream_direct.py": DIRECT_SHA256,
    "aiistream_prefetch.py": PREFETCH_SHA256,
}

class RuntimeHashError(RuntimeError):
    pass

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_runtime_hash(path: str | Path, expected: str = PROMOTED_RUNTIME_SHA256) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeHashError(
            f"runtime hash mismatch: expected {expected}, got {actual}"
        )
    return actual

def verify_runtime_imports(engine_dir: str | Path, mode: str, *,
                           p3_expected: str = PROMOTED_RUNTIME_SHA256) -> dict[str, str]:
    if mode not in ("prefetch", "direct", "parallel"):
        raise RuntimeHashError(f"invalid expert_read_path: {mode!r}")
    names = ["aiistream_model.py", "aiistream_parallel.py"]
    if mode in ("prefetch", "direct"):
        names.append("aiistream_direct.py")
    if mode == "prefetch":
        names.append("aiistream_prefetch.py")
    pins = dict(PINNED_RUNTIME_SHA256)
    pins["aiistream_parallel.py"] = p3_expected
    return {name: verify_runtime_hash(Path(engine_dir) / name, pins[name]) for name in names}

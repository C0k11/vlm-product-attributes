"""Access to the processed splits. The test split is frozen: its file hash is
pinned here and checked on every load, so any change to labels or membership
fails loudly instead of silently changing reported numbers."""
import hashlib
import json
from pathlib import Path

# Frozen 2026-09-22 after the val pilot; built by scripts/build_dataset.py at commit 6c82eb6
# (label maps unchanged since). 5,437 rows, 3,005 groups.
FROZEN_TEST_SHA256 = "8783fe3af0cebf61637ac4c8f0a76417d19c275f53b0eca966aa1313190d0f93"


def load_split(name: str, root: str | Path = "data/processed") -> list[dict]:
    path = Path(root) / f"{name}.jsonl"
    if name == "test":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != FROZEN_TEST_SHA256:
            raise RuntimeError(f"test split changed: sha256 {digest} != frozen {FROZEN_TEST_SHA256}")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]

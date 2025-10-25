"""Generate a SHA-256 file manifest for the portable build."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate file manifest")
    parser.add_argument("root", type=Path, help="Directory to hash")
    parser.add_argument("manifest", type=Path, help="Output manifest path")
    args = parser.parse_args()

    entries = []
    for file_path in iter_files(args.root):
        rel_path = file_path.relative_to(args.root)
        entries.append({
            "path": str(rel_path).replace("\\", "/"),
            "sha256": hash_file(file_path),
        })
    args.manifest.write_text(json.dumps({"files": entries}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

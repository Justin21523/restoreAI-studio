#!/usr/bin/env python3
"""Build and validate the static portfolio demo."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "portfolio-web"
DIST = ROOT / "dist"
REQUIRED = ["index.html", "styles.css", "app.js", "sample-data.json"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def build(check: bool = False) -> None:
    missing = [name for name in REQUIRED if not (SOURCE / name).exists()]
    if missing:
        raise SystemExit(f"Missing static demo files: {', '.join(missing)}")

    if not check:
        if DIST.exists():
            shutil.rmtree(DIST)
        DIST.mkdir(parents=True)
        for item in SOURCE.iterdir():
            if item.name.startswith("."):
                continue
            target = DIST / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    target_root = DIST if DIST.exists() else SOURCE
    for name in REQUIRED:
        target = target_root / name
        if not target.exists():
            raise SystemExit(f"Build output missing {name}")

    html = (target_root / "index.html").read_text(encoding="utf-8")
    for marker in ["RestorAI Studio", "demo-workbench", "before-after"]:
        if marker not in html:
            raise SystemExit(f"index.html missing marker: {marker}")

    print(
        {
            "mode": "check" if check else "build",
            "root": str(target_root),
            "index_sha": digest(target_root / "index.html"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate current build/source")
    args = parser.parse_args()
    build(check=args.check)


if __name__ == "__main__":
    main()

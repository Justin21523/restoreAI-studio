#!/usr/bin/env python3
"""Build or validate the canonical React portfolio demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DIST = ROOT / "dist"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def validate(root: Path) -> None:
    required = [root / "index.html"]
    if root == WEB:
        required.extend(
            [
                WEB / "src" / "App.tsx",
                WEB / "src" / "demo-scenarios.json",
                WEB / "src" / "demo-evidence.json",
                WEB / "public" / "demo" / "scenarios" / "archive-portrait" / "output.png",
                WEB / "public" / "demo" / "scenarios" / "archive-portrait" / "gfpgan.png",
                WEB / "public" / "demo" / "scenarios" / "archive-portrait" / "codeformer.png",
                WEB / "public" / "demo" / "scenarios" / "product-detail" / "output.png",
                WEB / "public" / "demo" / "scenarios" / "city-motion" / "output.mp4",
                WEB / "public" / "demo" / "scenarios" / "combined-video" / "output.mp4",
                ROOT / "package.json",
                ROOT / "vite.config.ts",
            ]
        )
    else:
        required.extend(
            [
                DIST / "assets",
                DIST / "demo" / "scenarios" / "archive-portrait" / "output.png",
                DIST / "demo" / "scenarios" / "product-detail" / "output.png",
                DIST / "demo" / "scenarios" / "city-motion" / "output.mp4",
                DIST / "demo" / "scenarios" / "combined-video" / "output.mp4",
            ]
        )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing web files: {', '.join(missing)}")
    html = (root / "index.html").read_text(encoding="utf-8")
    marker = "RestorAI Studio" if root == WEB else '<div id="root"></div>'
    if marker not in html:
        raise SystemExit(f"index.html missing marker: {marker}")


def build(check: bool = False) -> None:
    validate(WEB)
    if not check:
        subprocess.run(["npm", "run", "build"], cwd=ROOT, check=True)
        validate(DIST)
    root = DIST if DIST.exists() else WEB
    print(
        json.dumps(
            {
                "mode": "check" if check else "build",
                "root": str(root),
                "index_sha": digest(root / "index.html"),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build(check=args.check)


if __name__ == "__main__":
    main()

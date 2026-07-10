#!/usr/bin/env python3
"""Record a real API → RQ → CUDA → Artifact browser journey."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get("RESTORAI_DEMO_BASE_URL", "http://127.0.0.1:18765")
INPUT = ROOT / "web" / "public" / "demo" / "scenarios" / "product-detail" / "input.jpg"
OUTPUT = ROOT / "docs" / "demo" / "real-gpu-flow.webm"


def capture() -> None:
    if not INPUT.is_file():
        raise SystemExit(f"Missing input: {INPUT}")
    with (
        tempfile.TemporaryDirectory(prefix="restorai-real-video-") as temporary,
        sync_playwright() as playwright,
    ):
        executable = shutil.which("chromium") or shutil.which("chromium-browser")
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1050},
            record_video_dir=temporary,
            record_video_size={"width": 1440, "height": 1050},
        )
        page = context.new_page()
        page.goto(f"{BASE_URL}/#/workspace", wait_until="networkidle")
        page.wait_for_timeout(700)
        page.get_by_label("Preset").select_option("product")
        page.locator('input[type="file"]').set_input_files(str(INPUT))
        page.wait_for_timeout(800)
        page.get_by_role("button", name="Queue restoration").click()
        page.wait_for_url("**/#/jobs/**", timeout=20_000)
        page.get_by_text("succeeded", exact=True).wait_for(timeout=120_000)
        page.wait_for_timeout(1000)
        page.get_by_role("slider", name="Before and after position").fill("25")
        page.wait_for_timeout(600)
        page.get_by_role("slider", name="Before and after position").fill("75")
        page.wait_for_timeout(900)
        page.get_by_role("link", name="Jobs").click()
        page.wait_for_timeout(900)
        page.get_by_role("link", name="System").click()
        page.wait_for_timeout(1200)
        video = page.video
        context.close()
        if video is None:
            raise SystemExit("Playwright did not create a video")
        shutil.copy2(video.path(), OUTPUT)
        browser.close()
    print({"real_gpu_video": str(OUTPUT), "input": str(INPUT), "base_url": BASE_URL})


if __name__ == "__main__":
    capture()

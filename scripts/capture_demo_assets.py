#!/usr/bin/env python3
"""Capture portfolio screenshots and a short demo video for the static demo."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "dist"
OUT = ROOT / "docs" / "demo"
SCREENSHOTS = OUT / "screenshots"
VIDEO = OUT / "demo-tour.webm"
PORT = 8765


def wait_for_server(process: subprocess.Popen[bytes]) -> None:
    import urllib.request

    url = f"http://127.0.0.1:{PORT}/"
    for _ in range(40):
        if process.poll() is not None:
            raise SystemExit("Static demo server exited early")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise SystemExit("Timed out waiting for static demo server")


def convert_to_webp(source: Path, target: Path, quality: int = 86) -> None:
    image = Image.open(source).convert("RGB")
    image.save(target, "WEBP", quality=quality, method=6)
    source.unlink()


def capture() -> None:
    subprocess.run(["npm", "run", "build"], cwd=ROOT, check=True)
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    tmp_video = OUT / "_video"
    if tmp_video.exists():
        shutil.rmtree(tmp_video)
    tmp_video.mkdir(parents=True)

    server = subprocess.Popen(
        ["python", "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
        cwd=WEB,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(server)
        with sync_playwright() as p:
            executable = shutil.which("chromium") or shutil.which("chromium-browser")
            browser = p.chromium.launch(headless=True, executable_path=executable)

            page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
            page.screenshot(path=str(SCREENSHOTS / "showcase.png"), full_page=True)
            page.goto(
                f"http://127.0.0.1:{PORT}/#/workspace?scenario=face-lab",
                wait_until="networkidle",
            )
            page.screenshot(path=str(SCREENSHOTS / "workspace.png"), full_page=True)
            page.get_by_role("button", name="Run demo flow").click()
            page.wait_for_timeout(2300)
            page.screenshot(path=str(SCREENSHOTS / "pipeline-complete.png"), full_page=False)
            page.get_by_role("link", name="Inspect Job & Artifact").click()
            page.wait_for_timeout(500)
            page.locator("header").evaluate("element => element.style.position = 'static'")
            page.screenshot(path=str(SCREENSHOTS / "job-detail.png"), full_page=True)

            page.goto(f"http://127.0.0.1:{PORT}/#/batches/demo-batch-partial")
            page.screenshot(path=str(SCREENSHOTS / "batch-recovery.png"), full_page=True)

            mobile = browser.new_page(viewport={"width": 390, "height": 900}, is_mobile=True)
            mobile.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
            mobile.screenshot(path=str(SCREENSHOTS / "mobile.png"), full_page=True)

            context = browser.new_context(
                viewport={"width": 1440, "height": 1050},
                record_video_dir=str(tmp_video),
                record_video_size={"width": 1440, "height": 1050},
            )
            video_page = context.new_page()
            video_page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
            video_page.wait_for_timeout(700)
            video_page.get_by_role("link", name="Open interactive lab").click()
            video_page.wait_for_timeout(700)
            video_page.get_by_role("slider", name="Before and after position").fill("28")
            video_page.wait_for_timeout(500)
            video_page.get_by_role("slider", name="Before and after position").fill("72")
            video_page.wait_for_timeout(500)
            video_page.get_by_role("button", name="Run demo flow").click()
            video_page.wait_for_timeout(1800)
            video_page.get_by_role("link", name="Inspect Job & Artifact").click()
            video_page.wait_for_timeout(1000)
            video_page.get_by_role("link", name="Jobs").click()
            video_page.wait_for_timeout(700)
            video_page.get_by_text("Batch partial failure replay").click()
            video_page.wait_for_timeout(900)
            video_page.get_by_role("link", name="Workspace").click()
            video_page.wait_for_timeout(600)
            video_page.get_by_label("Demo scenario").select_option("product-detail")
            video_page.wait_for_timeout(900)
            video_page.get_by_role("link", name="Models").click()
            video_page.wait_for_timeout(1000)
            video_page.get_by_role("link", name="System").click()
            video_page.wait_for_timeout(1000)
            video_handle = video_page.video
            context.close()
            if video_handle is not None:
                recorded = Path(video_handle.path())
                shutil.copy2(recorded, VIDEO)

            browser.close()

        for png_name in [
            "showcase",
            "workspace",
            "pipeline-complete",
            "job-detail",
            "batch-recovery",
            "mobile",
        ]:
            convert_to_webp(SCREENSHOTS / f"{png_name}.png", SCREENSHOTS / f"{png_name}.webp")
        shutil.copy2(SCREENSHOTS / "showcase.webp", OUT / "cover.webp")
    finally:
        server.terminate()
        server.wait(timeout=5)
        if tmp_video.exists():
            shutil.rmtree(tmp_video)

    print(
        {
            "cover": str(OUT / "cover.webp"),
            "screenshots": sorted(path.name for path in SCREENSHOTS.glob("*.webp")),
            "video": str(VIDEO),
        }
    )


if __name__ == "__main__":
    capture()

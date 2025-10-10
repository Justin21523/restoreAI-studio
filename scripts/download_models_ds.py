#!/usr/bin/env python3
"""
Model download script for RestorAI
Downloads all required models to shared cache
"""
import argparse
import requests
import os
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.config import get_settings

settings = get_settings()


def download_file(url: str, destination: Path, description: str = "Downloading"):
    """Download file with progress bar"""
    destination.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as file, tqdm(
        desc=description,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_esrgan_models():
    """Download Real-ESRGAN models"""
    models = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    }

    for filename, url in models.items():
        dest_path = settings.paths["models"] / "enhancement" / "esrgan" / filename
        if dest_path.exists():
            print(f"✓ {filename} already exists")
            continue

        print(f"↓ Downloading {filename}...")
        download_file(url, dest_path, f"Downloading {filename}")


def download_gfpgan_models():
    """Download GFPGAN models"""
    models = {
        "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    }

    for filename, url in models.items():
        dest_path = settings.paths["models"] / "enhancement" / "gfpgan" / filename
        if dest_path.exists():
            print(f"✓ {filename} already exists")
            continue

        print(f"↓ Downloading {filename}...")
        download_file(url, dest_path, f"Downloading {filename}")


def main():
    parser = argparse.ArgumentParser(description="Download RestorAI models")
    parser.add_argument(
        "--model",
        choices=["esrgan", "gfpgan", "all"],
        default="all",
        help="Model type to download",
    )

    args = parser.parse_args()

    print("🔧 RestorAI Model Downloader")
    print(f"📁 Cache directory: {settings.paths['models']}")
    print()

    try:
        if args.model in ["esrgan", "all"]:
            print("📥 Downloading Real-ESRGAN models...")
            download_esrgan_models()
            print()

        if args.model in ["gfpgan", "all"]:
            print("📥 Downloading GFPGAN models...")
            download_gfpgan_models()
            print()

        print("✅ All models downloaded successfully!")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

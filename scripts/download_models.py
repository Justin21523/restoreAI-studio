# scripts/download_model.py
#!/usr/bin/env python3
"""
Model download script for RestorAI-studio .
Downloads AI models from official sources with progress tracking.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import requests
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model download URLs and info
MODELS = {
    "esrgan": {
        "name": "Real-ESRGAN x4plus",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "dir": "esrgan",
        "size_mb": 67.0,
        "description": "General purpose 4x super-resolution model",
    },
    "gfpgan": {
        "name": "GFPGAN v1.4",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "filename": "GFPGANv1.4.pth",
        "dir": "gfpgan",
        "size_mb": 348.0,
        "description": "Face restoration and enhancement",
    },
    "rife": {
        "name": "RIFE v4.6",
        "url": "https://github.com/megvii-research/ECCV2022-RIFE/releases/download/4.0/flownet.pkl",
        "filename": "flownet.pkl",
        "dir": "rife",
        "size_mb": 19.0,
        "description": "Video frame interpolation",
    },
}

# Essential models (smaller download)
ESSENTIAL_MODELS = ["esrgan"]


def download_file(url: str, output_path: Path, expected_size_mb: float = None) -> bool:  # type: ignore
    """Download file with progress bar."""
    try:
        logger.info(f"Downloading {output_path.name}...")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists
        if output_path.exists():
            current_size_mb = output_path.stat().st_size / (1024 * 1024)
            if expected_size_mb and abs(current_size_mb - expected_size_mb) < 1.0:
                logger.info(f"✅ {output_path.name} already exists and size matches")
                return True
            else:
                logger.info(f"📥 Re-downloading {output_path.name} (size mismatch)")

        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Verify download
        if output_path.exists():
            actual_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Downloaded {output_path.name} ({actual_size_mb:.1f} MB)")

            if expected_size_mb and abs(actual_size_mb - expected_size_mb) > 5.0:
                logger.warning(
                    f"⚠️ Size mismatch: expected {expected_size_mb:.1f} MB, got {actual_size_mb:.1f} MB"
                )

            return True
        else:
            logger.error(f"❌ Download failed: {output_path.name}")
            return False

    except requests.RequestException as e:
        logger.error(f"❌ Download failed for {output_path.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error downloading {output_path.name}: {e}")
        return False


def check_disk_space(required_mb: float) -> bool:
    """Check if enough disk space is available."""
    try:
        import shutil

        free_space = shutil.disk_usage(".").free / (1024 * 1024)

        if free_space < required_mb:
            logger.error(
                f"❌ Insufficient disk space: {free_space:.1f} MB available, {required_mb:.1f} MB required"
            )
            return False

        logger.info(f"💾 Disk space: {free_space:.1f} MB available")
        return True

    except Exception as e:
        logger.warning(f"⚠️ Could not check disk space: {e}")
        return True  # Assume it's okay if we can't check


def list_models():
    """List available models for download."""
    print("\n📦 Available Models:\n")

    for model_id, info in MODELS.items():
        essential = "⭐" if model_id in ESSENTIAL_MODELS else "  "
        print(
            f"{essential} {model_id:10} - {info['name']:20} ({info['size_mb']:6.1f} MB)"
        )
        print(f"   {' ' * 12} {info['description']}")

    print(f"\n⭐ Essential models (smaller download): {', '.join(ESSENTIAL_MODELS)}")
    print(f"💾 Total size: {sum(info['size_mb'] for info in MODELS.values()):.1f} MB")


def download_models(model_list: List[str], config: Config) -> Dict[str, bool]:
    """Download specified models."""
    results = {}

    # Calculate total size
    total_size_mb = sum(
        MODELS[model_id]["size_mb"] for model_id in model_list if model_id in MODELS
    )

    # Check disk space
    if not check_disk_space(total_size_mb * 1.2):  # 20% buffer
        return results

    logger.info(
        f"📥 Downloading {len(model_list)} models ({total_size_mb:.1f} MB total)"
    )

    for model_id in model_list:
        if model_id not in MODELS:
            logger.error(f"❌ Unknown model: {model_id}")
            results[model_id] = False
            continue

        model_info = MODELS[model_id]
        output_path = config.model_dir / model_info["dir"] / model_info["filename"]

        success = download_file(
            url=model_info["url"],
            output_path=output_path,
            expected_size_mb=model_info["size_mb"],
        )

        results[model_id] = success

    return results


def verify_models(config: Config) -> Dict[str, bool]:
    """Verify that downloaded models are valid."""
    logger.info("🔍 Verifying downloaded models...")

    results = {}
    for model_id, info in MODELS.items():
        model_path = config.model_dir / info["dir"] / info["filename"]

        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            expected_size = info["size_mb"]

            if abs(size_mb - expected_size) < 5.0:  # 5MB tolerance
                results[model_id] = True
                logger.info(f"✅ {info['name']} - OK ({size_mb:.1f} MB)")
            else:
                results[model_id] = False
                logger.warning(
                    f"⚠️ {info['name']} - Size mismatch ({size_mb:.1f} MB, expected {expected_size:.1f} MB)"
                )
        else:
            results[model_id] = False
            logger.info(f"❌ {info['name']} - Not found")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download AI models for RestorAI MVP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py --list              # List available models
  python download_models.py --essential         # Download essential models only
  python download_models.py --all               # Download all models
  python download_models.py esrgan gfpgan       # Download specific models
  python download_models.py --verify            # Verify existing models
        """,
    )

    parser.add_argument("models", nargs="*", help="Specific models to download")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument(
        "--essential", action="store_true", help="Download essential models only"
    )
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--verify", action="store_true", help="Verify existing models")
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )

    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.create_directories()

    if args.list:
        list_models()
        return

    if args.verify:
        results = verify_models(config)
        valid_count = sum(results.values())
        total_count = len(results)

        print(f"\n📊 Verification Summary: {valid_count}/{total_count} models valid")

        if valid_count == 0:
            print("💡 Run with --essential to download basic models")
        elif valid_count < total_count:
            missing = [model_id for model_id, valid in results.items() if not valid]
            print(
                f"💡 Download missing models: python download_models.py {' '.join(missing)}"
            )

        return

    # Determine which models to download
    if args.all:
        models_to_download = list(MODELS.keys())
    elif args.essential:
        models_to_download = ESSENTIAL_MODELS
    elif args.models:
        models_to_download = args.models
    else:
        # Interactive mode
        print("🤖 RestorAI MVP - Model Downloader")
        list_models()

        choice = input("\nEnter choice (essential/all/model names): ").strip().lower()

        if choice == "essential":
            models_to_download = ESSENTIAL_MODELS
        elif choice == "all":
            models_to_download = list(MODELS.keys())
        else:
            models_to_download = choice.split()

    # Download models
    if not models_to_download:
        logger.error("❌ No models specified for download")
        return

    logger.info(f"🚀 Starting download of models: {', '.join(models_to_download)}")
    results = download_models(models_to_download, config)

    # Summary
    successful = [model for model, success in results.items() if success]
    failed = [model for model, success in results.items() if not success]

    print(f"\n📊 Download Summary:")
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    if successful:
        print(f"   {', '.join(successful)}")

    if failed:
        print(f"❌ Failed: {len(failed)}")
        print(f"   {', '.join(failed)}")
        print(
            f"\n💡 Try downloading failed models individually or check your internet connection"
        )

    if successful:
        print(f"\n🎉 Ready to use! Launch with: python run.py --mode ui")


if __name__ == "__main__":
    main()

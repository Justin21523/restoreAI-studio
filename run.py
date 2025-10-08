#!/usr/bin/env python3
"""
RestorAI MVP - Single Entry Point
A lightweight, portable AI image restoration tool.

Usage:
    python run.py --mode ui          # Launch Gradio interface
    python run.py --mode api         # Launch REST API server
    python run.py --mode cli         # Command line interface
    python run.py --help             # Show help
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from ui.cli.main import CLIApp

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment and create necessary directories."""
    # Create data directories
    directories = [
        "data/input",
        "data/output",
        "data/temp",
        "data/models/esrgan",
        "data/models/gfpgan",
        "data/models/rife",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        from utils.config import load_env

        load_env(env_file)

    logger.info("Environment setup completed")


def launch_gradio_ui(share=False, debug=False):
    """Launch Gradio user interface."""
    try:
        from ui.gradio.app import create_gradio_app

        logger.info("Starting Gradio interface...")
        app = create_gradio_app()

        port = int(os.getenv("UI_PORT", "7860"))
        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            debug=debug,
            show_api=False,
        )

    except ImportError as e:
        logger.error(f"Failed to import Gradio components: {e}")
        logger.error(
            "Please install required dependencies: pip install -r requirements.txt"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start Gradio interface: {e}")
        sys.exit(1)


def launch_api_server(host="0.0.0.0", port=8000, reload=False):
    """Launch FastAPI server."""
    try:
        import uvicorn
        from api.main import app

        logger.info(f"Starting API server at http://{host}:{port}")
        logger.info(f"API documentation: http://{host}:{port}/docs")

        uvicorn.run(
            "api.main:app", host=host, port=port, reload=reload, log_level="info"
        )

    except ImportError as e:
        logger.error(f"Failed to import API components: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)


def launch_cli(args):
    """Launch command line interface."""
    try:
        cli_app = CLIApp()
        cli_app.run(args)

    except ImportError as e:
        logger.error(f"Failed to import CLI components: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        sys.exit(1)


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
    ]

    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(name)

    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.error("Please install: pip install -r requirements.txt")
        return False

    return True


def check_models():
    """Check if AI models are available."""
    model_paths = {
        "Real-ESRGAN": "data/models/esrgan/RealESRGAN_x4plus.pth",
        "GFPGAN": "data/models/gfpgan/GFPGANv1.4.pth",
        "RIFE": "data/models/rife/flownet.pkl",
    }

    available_models = []
    for name, path in model_paths.items():
        if Path(path).exists():
            available_models.append(name)

    if available_models:
        logger.info(f"Available models: {', '.join(available_models)}")
    else:
        logger.warning(
            "No AI models found. Download with: python scripts/download_models.py"
        )

    return len(available_models) > 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RestorAI MVP - AI Image Restoration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode ui                    # Launch web interface
  python run.py --mode api --port 8080       # Launch API on port 8080
  python run.py --mode cli upscale image.jpg # Upscale image via CLI
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["ui", "api", "cli"],
        default="ui",
        help="Launch mode (default: ui)",
    )

    # UI options
    parser.add_argument(
        "--share", action="store_true", help="Share Gradio interface publicly"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # API options
    parser.add_argument("--host", default="0.0.0.0", help="API host (default: 0.0.0.0)")
    parser.add_argument(
        "--port", type=int, default=8000, help="API port (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    # CLI options
    parser.add_argument(
        "command", nargs="?", help="CLI command (upscale, face-restore, etc.)"
    )
    parser.add_argument("input_file", nargs="?", help="Input file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument(
        "--scale", type=int, default=4, help="Upscale factor (default: 4)"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Restoration strength (default: 0.8)",
    )

    args = parser.parse_args()

    # Setup environment
    setup_environment()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check models (warning only)
    check_models()

    # Launch appropriate mode
    if args.mode == "ui":
        launch_gradio_ui(share=args.share, debug=args.debug)

    elif args.mode == "api":
        launch_api_server(host=args.host, port=args.port, reload=args.reload)

    elif args.mode == "cli":
        if not args.command:
            parser.error("CLI mode requires a command")
        launch_cli(args)

    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

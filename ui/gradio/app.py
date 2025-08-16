# ui/gradio/app.py
"""
RestorAI MVP - Gradio Interface
Simple, user-friendly web interface for AI image restoration.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
import gradio as gr
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.config import Config
from core.esrgan import ESRGANProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = Config()
config.create_directories()

# Global processors (loaded on demand)
processors = {}


def get_processor(model_name: str):
    """Get or create processor for the specified model."""
    if model_name not in processors:
        try:
            if model_name == "esrgan":
                if config.esrgan_model_path.exists():
                    processors[model_name] = ESRGANProcessor(
                        model_path=config.esrgan_model_path, device=config.device
                    )
                else:
                    raise FileNotFoundError(
                        f"Model not found: {config.esrgan_model_path}"
                    )

            # Add other processors here as implemented
            # elif model_name == "gfpgan":
            #     processors[model_name] = GFPGANProcessor(...)

            else:
                raise ValueError(f"Unknown model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    return processors[model_name]


def upscale_image(
    image: np.ndarray, scale: int, model: str
) -> Tuple[Optional[np.ndarray], str]:
    """
    Upscale image using the specified model.

    Args:
        image: Input image as numpy array
        scale: Upscale factor (2, 4, 8)
        model: Model to use ("esrgan")

    Returns:
        Tuple of (processed_image, status_message)
    """
    if image is None:
        return None, "❌ Please upload an image first"

    try:
        # Check image size
        h, w = image.shape[:2]
        if max(h, w) > config.max_image_size:
            return None, f"❌ Image too large. Max size: {config.max_image_size}px"

        # Get processor
        processor = get_processor(model)

        # Process image
        logger.info(f"Processing image with {model}, scale {scale}x")
        result, processing_time = processor.process_with_timing(image, scale=scale)

        # Convert result to proper format
        if isinstance(result, np.ndarray):
            result = np.clip(result, 0, 255).astype(np.uint8)

        status = f"✅ Processed successfully in {processing_time:.1f}s (Scale: {scale}x, Model: {model})"
        return result, status

    except FileNotFoundError as e:
        return None, f"❌ Model not found: {str(e)}"
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return None, f"❌ Processing failed: {str(e)}"


def face_restore_image(
    image: np.ndarray, strength: float, model: str
) -> Tuple[Optional[np.ndarray], str]:
    """
    Restore faces in image.

    Args:
        image: Input image as numpy array
        strength: Restoration strength (0.0-1.0)
        model: Model to use ("gfpgan")

    Returns:
        Tuple of (processed_image, status_message)
    """
    if image is None:
        return None, "❌ Please upload an image first"

    try:
        # This is a placeholder - implement when GFPGAN processor is ready
        return None, f"🚧 Face restoration not yet implemented (coming soon!)"

    except Exception as e:
        logger.error(f"Face restoration failed: {e}")
        return None, f"❌ Face restoration failed: {str(e)}"


def get_model_status() -> str:
    """Get status of available models."""
    models = config.get_model_info()

    status_lines = ["📊 **Model Status:**"]
    for name, info in models.items():
        status = "✅" if info["available"] else "❌"
        size_info = f" ({info['size_mb']:.1f} MB)" if info["available"] else ""
        status_lines.append(f"- {status} {info['name']}{size_info}")

    status_lines.append(f"\n🖥️ **Device:** {config.device}")
    status_lines.append(f"📏 **Max Image Size:** {config.max_image_size}px")

    return "\n".join(status_lines)


def create_gradio_app() -> gr.Blocks:
    """Create and configure Gradio interface."""

    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto;
    }
    .status-info {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
    }
    """

    with gr.Blocks(title="RestorAI MVP", css=css, theme=gr.themes.Soft()) as app:

        # Header
        gr.Markdown(
            """
        # 🎨 RestorAI MVP
        ### Lightweight AI Image Restoration Tool

        Upload an image and enhance it using state-of-the-art AI models.
        """
        )

        # Model status
        with gr.Row():
            status_display = gr.Markdown(
                get_model_status(), elem_classes=["status-info"]
            )
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")

        refresh_btn.click(fn=get_model_status, outputs=status_display)

        # Main interface
        with gr.Tabs():

            # Super Resolution Tab
            with gr.TabItem("🔍 Super Resolution"):
                with gr.Row():
                    with gr.Column():
                        sr_input = gr.Image(
                            label="Input Image", type="numpy", height=400
                        )

                        with gr.Row():
                            sr_scale = gr.Dropdown(
                                choices=[2, 4, 8], value=4, label="Upscale Factor"
                            )
                            sr_model = gr.Dropdown(
                                choices=["esrgan"], value="esrgan", label="Model"
                            )

                        sr_button = gr.Button("🚀 Upscale Image", variant="primary")

                    with gr.Column():
                        sr_output = gr.Image(
                            label="Enhanced Image", type="numpy", height=400
                        )
                        sr_status = gr.Textbox(
                            label="Status", interactive=False, max_lines=3
                        )

                sr_button.click(
                    fn=upscale_image,
                    inputs=[sr_input, sr_scale, sr_model],
                    outputs=[sr_output, sr_status],
                )

            # Face Restoration Tab (Placeholder)
            with gr.TabItem("👤 Face Restoration"):
                with gr.Row():
                    with gr.Column():
                        face_input = gr.Image(
                            label="Input Image", type="numpy", height=400
                        )

                        with gr.Row():
                            face_strength = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                                label="Restoration Strength",
                            )
                            face_model = gr.Dropdown(
                                choices=["gfpgan"], value="gfpgan", label="Model"
                            )

                        face_button = gr.Button("✨ Restore Faces", variant="primary")

                    with gr.Column():
                        face_output = gr.Image(
                            label="Restored Image", type="numpy", height=400
                        )
                        face_status = gr.Textbox(
                            label="Status", interactive=False, max_lines=3
                        )

                face_button.click(
                    fn=face_restore_image,
                    inputs=[face_input, face_strength, face_model],
                    outputs=[face_output, face_status],
                )

        # Footer
        gr.Markdown(
            """
        ---
        💡 **Tips:**
        - For best results, use clear, high-quality input images
        - Larger images may take longer to process
        - Try different scale factors to find the best balance between quality and file size

        📚 **Need help?** Check the documentation or report issues on GitHub.
        """
        )

    return app


def main():
    """Main function to launch Gradio app."""
    try:
        app = create_gradio_app()

        # Launch configuration
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": config.ui_port,
            "share": config.ui_share,
            "show_api": False,
            "quiet": False,
        }

        logger.info(f"Starting Gradio interface on port {config.ui_port}")
        app.launch(**launch_kwargs)

    except Exception as e:
        logger.error(f"Failed to start Gradio app: {e}")
        raise


if __name__ == "__main__":
    main()

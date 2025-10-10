import gradio as gr
import requests
import tempfile
import os
from pathlib import Path
import json

# API configuration
API_BASE = "http://localhost:8000/api/v1"


def super_resolution_interface(image, scale, model):
    """Gradio interface for super-resolution"""
    if image is None:
        return None, "Please upload an image"

    # Save image to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)

        # Call API
        files = {"file": (tmp_file.name, open(tmp_file.name, "rb"), "image/png")}
        data = {"scale": scale, "model": model}

        try:
            response = requests.post(f"{API_BASE}/restore/sr", files=files, data=data)
            response.raise_for_status()

            result = response.json()
            output_url = result["url"]

            # Download result
            output_response = requests.get(output_url)
            output_path = tempfile.mktemp(suffix=".png")
            with open(output_path, "wb") as f:
                f.write(output_response.content)

            # Cleanup
            os.unlink(tmp_file.name)

            return output_path, f"Success! Processed in {result['latency_ms']:.0f}ms"

        except Exception as e:
            os.unlink(tmp_file.name)
            return None, f"Error: {str(e)}"


def face_restoration_interface(image, method, strength):
    """Gradio interface for face restoration"""
    if image is None:
        return None, "Please upload an image"

    # Save image to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)

        # Call API
        files = {"file": (tmp_file.name, open(tmp_file.name, "rb"), "image/png")}
        data = {"method": method, "strength": strength}

        try:
            response = requests.post(f"{API_BASE}/restore/face", files=files, data=data)
            response.raise_for_status()

            result = response.json()
            output_url = result["url"]

            # Download result
            output_response = requests.get(output_url)
            output_path = tempfile.mktemp(suffix=".png")
            with open(output_path, "wb") as f:
                f.write(output_response.content)

            # Cleanup
            os.unlink(tmp_file.name)

            return (
                output_path,
                f"Success! Found {result['face_count']} faces in {result['latency_ms']:.0f}ms",
            )

        except Exception as e:
            os.unlink(tmp_file.name)
            return None, f"Error: {str(e)}"


def create_gradio_interface():
    """Create Gradio interface with multiple tabs"""
    with gr.Blocks(
        title="RestorAI - Image Restoration", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# 🖼️ RestorAI - AI Image Restoration")
        gr.Markdown("Enhance your images with super-resolution and face restoration")

        with gr.Tab("Super Resolution"):
            with gr.Row():
                with gr.Column():
                    sr_input = gr.Image(label="Input Image", type="pil")
                    sr_scale = gr.Radio([2, 4], label="Scale Factor", value=2)
                    sr_model = gr.Radio(
                        ["realesrgan-x2plus", "realesrgan-x4plus"],
                        label="Model",
                        value="realesrgan-x4plus",
                    )
                    sr_button = gr.Button("Enhance Resolution", variant="primary")

                with gr.Column():
                    sr_output = gr.Image(label="Enhanced Image")
                    sr_status = gr.Textbox(label="Status", interactive=False)

            sr_button.click(
                super_resolution_interface,
                inputs=[sr_input, sr_scale, sr_model],
                outputs=[sr_output, sr_status],
            )

        with gr.Tab("Face Restoration"):
            with gr.Row():
                with gr.Column():
                    face_input = gr.Image(label="Input Image", type="pil")
                    face_method = gr.Radio(
                        ["gfpgan", "codeformer"], label="Method", value="gfpgan"
                    )
                    face_strength = gr.Slider(
                        0, 1, value=0.5, label="Restoration Strength"
                    )
                    face_button = gr.Button("Restore Faces", variant="primary")

                with gr.Column():
                    face_output = gr.Image(label="Restored Image")
                    face_status = gr.Textbox(label="Status", interactive=False)

            face_button.click(
                face_restoration_interface,
                inputs=[face_input, face_method, face_strength],
                outputs=[face_output, face_status],
            )

        with gr.Tab("About"):
            gr.Markdown(
                """
            ## About RestorAI

            **Features:**
            - 🚀 Super Resolution with Real-ESRGAN
            - 👤 Face Restoration with GFPGAN/CodeFormer
            - ⚡ Fast inference with GPU acceleration
            - 💾 Shared model cache system

            **How to use:**
            1. Upload an image in either tab
            2. Adjust parameters as needed
            3. Click the process button
            4. Download your enhanced image!
            """
            )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_port=7860, share=False)

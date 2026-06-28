from __future__ import annotations

import io
import time

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageFilter


app = FastAPI(
    title="RestorAI Studio (Demo API)",
    version="0.1.0",
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "mode": "demo", "note": "This demo uses Pillow resize (no Real-ESRGAN/GFPGAN)."}


def _read_image(file: UploadFile) -> Image.Image:
    try:
        raw = file.file.read()
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


@app.post("/api/v1/upscale")
def upscale(
    file: UploadFile = File(..., description="Input image"),
    scale: int = Form(default=2),
    method: str = Form(default="bicubic"),
    sharpen: bool = Form(default=True),
):
    """
    Lightweight image upscale demo:
    - uses Pillow resampling + optional unsharp mask
    - returns PNG
    """
    t0 = time.perf_counter()
    img = _read_image(file)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    scale = int(scale or 2)
    if scale not in (2, 3, 4, 6, 8):
        raise HTTPException(status_code=400, detail="scale must be one of: 2,3,4,6,8")

    method = (method or "bicubic").strip().lower()
    resample = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }.get(method)
    if resample is None:
        raise HTTPException(status_code=400, detail="method must be one of: nearest,bilinear,bicubic,lanczos")

    w, h = img.size
    out = img.resize((w * scale, h * scale), resample=resample)
    if bool(sharpen):
        out = out.filter(ImageFilter.UnsharpMask(radius=1.6, percent=160, threshold=3))

    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    payload = buf.getvalue()

    return Response(
        content=payload,
        media_type="image/png",
        headers={
            "X-Input-Size": f"{w}x{h}",
            "X-Output-Size": f"{w*scale}x{h*scale}",
            "X-Elapsed-Ms": str(int((time.perf_counter() - t0) * 1000)),
        },
    )


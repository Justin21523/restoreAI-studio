# API

## Stable Demo API

`webapp/main.py` exposes the CI-safe demo API.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Health check with `mode: demo`. |
| `POST` | `/api/v1/upscale` | Accepts an image and returns a resized/sharpened PNG. |

```bash
uvicorn webapp.main:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/api/v1/health
```

```bash
curl -F "file=@sample.png" \
  -F "scale=2" \
  -F "method=bicubic" \
  -F "sharpen=true" \
  http://127.0.0.1:8000/api/v1/upscale --output out.png
```

## Prototype Full API

The original `api/` package contains the broader FastAPI service scaffold: restore routes, jobs, batch, video, safety, history, metrics, exports, and admin endpoints. It is preserved as architecture evidence, but the stable public demo does not depend on it.

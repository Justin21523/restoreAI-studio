# Quickstart

Use the mock-safe path for interviews and CI.

```bash
pip install -r docker/requirements.demo.txt pytest httpx playwright
python -m pytest -q
python scripts/smoke_demo.py
python scripts/build_static_demo.py
python -m http.server 8080 -d dist
```

Open `http://127.0.0.1:8080`.

For the local demo API:

```bash
uvicorn webapp.main:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/api/v1/health
```

The full AI model path remains a prototype scaffold and requires model weights plus dependency hardening.

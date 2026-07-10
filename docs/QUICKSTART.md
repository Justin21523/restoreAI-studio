# Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu,dev]'
cp .env.example .env
npm ci && npm run build
docker compose up -d postgres redis
alembic upgrade head
restorai models verify
```

Terminal 1:

```bash
restorai-api
```

Terminal 2:

```bash
restorai-worker
```

Open `http://127.0.0.1:8000`. Run `pytest -q`, `ruff check restorai tests alembic
scripts`, and `npm run typecheck` before committing.

Build only the public simulation with:

```bash
VITE_APP_MODE=demo npm run build
python -m http.server 8080 -d dist
```

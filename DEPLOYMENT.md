# Deployment

## Local real environment

PostgreSQL and Redis run through `compose.yaml` on loopback ports 55432 and 56379.
The FastAPI process and one CUDA worker run on the host so PyTorch can use the
installed RTX 5080/CUDA stack without rebuilding a large GPU image.

```bash
docker compose up -d postgres redis
alembic upgrade head
VITE_APP_MODE=real npm run build
restorai-api
restorai-worker
```

Mount or preserve `data/storage` for in-flight uploads/artifacts. Back up PostgreSQL
for durable Job metadata. Redis is dispatch infrastructure, not the source of truth.

## API image

The root Dockerfile builds the React real-mode bundle and CPU control-plane API.
It does not package model weights or a CUDA worker. Mount `MODEL_ROOT` read-only and
`STORAGE_ROOT` read-write if using the image in a larger deployment.

## Public demo

`.github/workflows/pages.yml` builds `VITE_APP_MODE=demo`, uploads `dist/` as the
official Pages artifact, and deploys with GitHub's Pages Actions. The public demo
has no API, credentials, user uploads, GPU, or model weights.

## Release checklist

1. Run full model SHA verification.
2. Apply `alembic upgrade head` and check `/api/v1/health/ready`.
3. Run Python tests/lint and React typecheck/build.
4. Run `python scripts/gpu_smoke.py` and regenerate Demo results when models change.
5. Submit one real image Job and one short video Job through API/RQ.
6. Verify output dimensions, video FPS/audio, model snapshots, and downloads.
7. Confirm expiry cleanup and available disk space.
8. Review CodeFormer and other upstream license obligations for the target use.

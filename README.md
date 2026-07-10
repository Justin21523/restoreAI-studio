# RestorAI Studio

Local-first GPU workspace for restoring images and video with Real-ESRGAN,
GFPGAN, CodeFormer, and RIFE. Processing is durable and observable: uploads
become queued Jobs, progress is recorded as events, and successful results are
published as expiring Artifacts.

[Interactive GPU evidence demo](https://justin21523.github.io/restoreAI-studio/) ·
[Real API → RQ → CUDA recording](docs/demo/real-gpu-flow.webm) ·
[API reference](docs/API.md) · [Model registry](docs/MODELS.md) ·
[Deployment guide](DEPLOYMENT.md)

![RestorAI Studio workspace](docs/demo/cover.webp)

## What it solves

AI restoration tools often stop at a one-off notebook or synchronous upload
endpoint. RestorAI Studio packages the same inference into a product-shaped
workflow with strict local model ownership, GPU back-pressure, batch submission,
durable history, reproducible output metadata, cancellation, retry, and expiry.

## Current capabilities

| Capability | Implementation status |
| --- | --- |
| Image super-resolution | Real-ESRGAN x2plus and x4plus, CUDA validated |
| Face restoration | GFPGAN v1.4 and CodeFormer, face detection/parsing, CUDA validated |
| Video frame interpolation | RIFE v4.25 with source audio preservation, CUDA validated |
| Video super-resolution | Per-frame Real-ESRGAN x2/x4 export to H.264 MP4 |
| Combined video flow | RIFE interpolation followed by Real-ESRGAN upscale |
| Batch processing | Multi-file image/video API and directory CLI submission |
| Job system | PostgreSQL source of truth, Redis/RQ GPU queue, progress events, cancel/retry |
| Artifacts | SHA-256 input/output provenance, model snapshot, download/delete, 24-hour file expiry |
| Web product | Bilingual React workspace, presets, Job/Batch detail, comparison, Models and System views |
| Public demo | Four real GPU workflows, full Job/Batch replay, benchmarks and provenance on GitHub Pages |

## Architecture

```mermaid
flowchart LR
  UI[React workspace] --> API[FastAPI /api/v1]
  CLI[Directory CLI] --> API
  API --> PG[(PostgreSQL)]
  API --> OUTBOX[Transactional outbox]
  OUTBOX --> REDIS[(Redis / RQ)]
  REDIS --> WORKER[Single-concurrency GPU worker]
  WORKER --> PIPE[Image and video pipelines]
  PIPE --> REG[Strict model registry]
  REG --> ROOT[/mnt/c/ai_models]
  WORKER --> FS[(Local artifact storage)]
  WORKER --> PG
  API --> FS
```

The API does not perform inference. It persists a Job and outbox record first,
then dispatches work to a queue. A single worker owns the GPU to prevent VRAM
contention. PostgreSQL remains authoritative even if Redis is restarted.

## Model policy

Every weight is resolved from the absolute root `/mnt/c/ai_models`. Runtime
downloads are disabled. `restorai/model_manifest.yaml` pins each relative path,
minimum size, upstream source, and SHA-256 digest.

```bash
# Full integrity verification
python -m restorai.cli models verify

# Explicit, operator-initiated installation of missing manifest files only
python -m restorai.cli models install-missing
```

No weight is copied into the repository or runtime storage. See
[docs/MODELS.md](docs/MODELS.md) for the exact layout and licensing caveats.

## Local quick start

Requirements: Python 3.10, Node.js 22, Docker Compose, FFmpeg, an NVIDIA GPU,
and a CUDA-compatible PyTorch installation. This machine was validated with an
RTX 5080, PyTorch 2.9.1 + CUDA 13.0.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu,dev]'
cp .env.example .env

npm ci
npm run build

docker compose up -d postgres redis
alembic upgrade head
python -m restorai.cli models verify
```

Start the API and GPU worker in separate terminals:

```bash
restorai-api
restorai-worker
```

Open `http://127.0.0.1:8000`. Interactive API documentation is available at
`http://127.0.0.1:8000/api/v1/docs`.

For a batch directory submission:

```bash
restorai process-directory ./samples --operation upscale --pattern '*.png'
```

## API examples

```bash
curl -F 'file=@portrait.png' \
  -F 'operation=face_restore_upscale' \
  -F 'face_method=codeformer' \
  -F 'fidelity=0.7' \
  -F 'scale=2' \
  http://127.0.0.1:8000/api/v1/jobs/images

curl -F 'file=@clip.mp4' \
  -F 'operation=interpolate_upscale' \
  -F 'target_fps=60' \
  -F 'scale=2' \
  http://127.0.0.1:8000/api/v1/jobs/videos
```

Use `GET /api/v1/jobs/{id}/events` for Server-Sent Events and download the
result from `GET /api/v1/artifacts/{id}/download`.

## Repository map

```text
restorai/
  adapters/          model-specific inference adapters
  pipelines/         image/video validation and orchestration
  services/          job, storage, retention, and API client services
  vendor/            pinned inference architecture snapshots only
  api.py              versioned HTTP contract and SSE
  worker.py           RQ GPU worker
  model_registry.py   strict manifest/path/checksum boundary
  orm.py              durable domain schema
web/                  canonical React + TypeScript frontend
alembic/              PostgreSQL schema migrations
tests/unit/           API, registry, pipeline, job, error, and retention tests
tests/e2e/            Playwright public-demo and mocked real-mode journeys
scripts/              demo generation, GPU smoke, build, and reproducible capture
docs/                 operation and design documentation
```

The pre-refactor `api/`, `core/`, `ui/`, `utils/`, `webapp/`, Gradio, and PyQt
implementations were removed. They represented overlapping products with
incompatible configuration and incomplete model paths.

## Quality checks

```bash
pytest -q
ruff check restorai tests alembic scripts
npm test
npm run typecheck
npm run build
npm run e2e
python scripts/gpu_smoke.py
python scripts/build_demo_scenarios.py --gpu
alembic upgrade head --sql >/tmp/restorai-schema.sql
```

GPU inference is intentionally separated from unit tests. Run the explicit model
verification and a small real image/video Job before release. The validated
end-to-end path includes API upload → PostgreSQL/outbox → Redis/RQ → CUDA worker
→ Artifact download, including video FPS and audio-stream checks.

## Operational behavior

- GPU queue concurrency is one worker process by design.
- Failed work is recorded in both PostgreSQL and RQ's failed-job registry.
- Queued and running work can be cancelled; failed/cancelled Jobs can be retried.
- Uploaded and output files expire after 24 hours by default; metadata is retained.
- Image size/pixel count and video size/duration/resolution/FPS are bounded.
- Artifact downloads are constrained to `STORAGE_ROOT` to prevent path traversal.
- Health endpoints distinguish process liveness from database/Redis/model readiness.
- System status reports the RQ worker, queue depth, GPU/VRAM, storage, models, and services.
- Individual retries are detached from their original Batch; failed Batch retries create linked Batches.

## Public demo versus real mode

| Environment | Purpose | Models/data |
| --- | --- | --- |
| GitHub Pages | Fast interview walkthrough and responsive UX | Real GFPGAN, CodeFormer, Real-ESRGAN and RIFE outputs with workflow replay |
| Local real mode | Complete CUDA inference and Job lifecycle | `/mnt/c/ai_models`, PostgreSQL, Redis, local storage |
| API container | Portable control plane/frontend | Mount model/storage paths; GPU worker remains host-first |

Set `VITE_APP_MODE=demo` for Pages and `VITE_APP_MODE=real` for the local API
build. The visual treatment is shared, while the environment badge makes the
boundary explicit.

### Public demo scenarios

| Scenario | Real pipeline | Measured result on RTX 5080 |
| --- | --- | --- |
| Face restoration lab | GFPGAN / CodeFormer / Real-ESRGAN 2× | Same portrait, 1 face, 256² → 512² |
| Product detail | Real-ESRGAN 4× | 256² → 1024² with FP16 tiled inference |
| City motion | RIFE v4.25 | 24 → 48 FPS, 144 output frames, audio preserved |
| Combined video | RIFE + Real-ESRGAN 2× | 192×108/12 FPS → 384×216/24 FPS, audio preserved |

The public site also exposes successful and failed Job detail, Artifact SHA-256,
Batch partial failure/retry lineage, all seven model registry entries, a recorded
system snapshot, and cold/warm benchmark evidence. The project-owned masters,
generation notes, deterministic degradation, and GPU output process are documented
in [docs/demo/SOURCES.md](docs/demo/SOURCES.md).

The checked-in [real GPU browser recording](docs/demo/real-gpu-flow.webm) was
captured against the actual FastAPI/PostgreSQL/Redis/RQ stack. Its upload completed
as a real `upscale` Job with eight persisted events and a Real-ESRGAN x2plus
Artifact; it is separate from the transparent static workflow replay.

## Key engineering decisions

- **Manifest over discovery:** exact model IDs and hashes are auditable; arbitrary
  files under the warehouse cannot silently become production dependencies.
- **PostgreSQL plus outbox:** Job creation survives transient Redis failures and
  can be dispatched later without losing user intent.
- **One GPU owner:** predictable VRAM use is more important than misleading local
  parallelism on a single 16 GB card.
- **RIFE before upscale:** interpolation runs on fewer pixels, reducing latency and
  VRAM; Real-ESRGAN then enhances all source and generated frames consistently.
- **Files expire, metadata stays:** local disk remains bounded while portfolio and
  debugging evidence—parameters, hashes, timings, model versions—remains useful.
- **No CUDA initialization in the RQ parent:** GPU capability data comes from
  `nvidia-smi`, leaving each forked child free to initialize CUDA safely.

## Known constraints and next steps

- Authentication, quotas, malware scanning, object storage, and multi-node GPU
  scheduling are out of scope for the local portfolio release.
- RIFE target FPS is quantized to an integer multiple of source FPS.
- Large video jobs currently materialize PNG frames in a temporary work directory;
  a streaming/segmented pipeline is the next major performance improvement.
- CodeFormer uses the NTU S-Lab License 1.0; review its terms before any commercial
  or hosted production use.
- Add a larger quality benchmark corpus and objective image/video quality metrics.

## Portfolio demo script

1. Open Showcase; identify the four CUDA-validated model families and RTX 5080 evidence.
2. Use Face Lab to compare GFPGAN, CodeFormer, and the complete 2× pipeline.
3. Run a recorded flow and inspect its Job timeline, Artifact hashes and model snapshot.
4. Open the partial-failure Batch and replay failed-item retry lineage.
5. Compare RIFE 24/48 FPS and the RIFE → Real-ESRGAN combined video with audio.
6. Open Models and System to show all seven checksums and the recorded local stack.
7. Close with the outbox, one-GPU-owner and provenance architecture decisions.

Use non-sensitive sample data and keep clips under five seconds for a reliable
three-minute interview walkthrough. A timed narration is provided in
[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

## License and attribution

Project-specific source is intended for portfolio use. Vendored model architecture
files and weights remain governed by their upstream licenses. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution or hosted use.

<!-- portfolio-release-notes:start -->
## Portfolio release status

- Real CUDA adapters: implemented and locally validated.
- Durable API/queue/history/artifacts: implemented and end-to-end validated.
- React Showcase, full workflow replay and GitHub Pages demo: implemented.
- Automated suite: Python unit/integration, React component, and Playwright journeys.
- Real GPU release smoke: GFPGAN, CodeFormer, Real-ESRGAN, RIFE, and combined video.
- Remaining production work: authentication, object storage, and multi-node scheduling.
<!-- portfolio-release-notes:end -->

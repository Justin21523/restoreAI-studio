# API reference

Base path: `/api/v1`. OpenAPI UI: `/api/v1/docs`.

## Health and models

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health/live` | Process/version liveness |
| GET | `/health/ready` | PostgreSQL, Redis, and fast model readiness |
| GET | `/models` | Read-only manifest status; never triggers downloads |

## Jobs and batches

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/jobs/images` | Queue one image Job |
| POST | `/jobs/videos` | Queue one video Job |
| POST | `/batches/images` | Queue 1–50 image Jobs |
| POST | `/batches/videos` | Queue 1–10 video Jobs |
| GET | `/jobs` | History, optionally filtered by status |
| GET | `/jobs/{id}` | Job state, events, nested Artifact, model provenance |
| GET | `/jobs/{id}/input` | Stream the unexpired original input |
| GET | `/jobs/{id}/events` | SSE progress stream; accepts `Last-Event-ID` |
| POST | `/jobs/{id}/cancel` | Cancel queued/running work |
| POST | `/jobs/{id}/retry` | Clone failed/cancelled Job as a standalone retry |
| GET | `/batches/{id}` | Aggregate status and all child Jobs |
| POST | `/batches/{id}/cancel` | Cancel all queued/running child Jobs |
| POST | `/batches/{id}/retry-failed` | Create a linked Batch for failed children only |
| GET | `/system/status` | Database, Redis, queue, worker, GPU, storage, and model status |

Image operations are `upscale`, `face_restore`, and `face_restore_upscale`.
Video operations are `interpolate`, `upscale`, and `interpolate_upscale`.

## Artifacts

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/artifacts/{id}` | Metadata, hashes, model snapshot, expiry |
| GET | `/artifacts/{id}/download` | Stream an unexpired output file |
| DELETE | `/artifacts/{id}` | Delete input/output file now; retain metadata |

Errors use standard FastAPI JSON responses. Worker failures populate stable codes:
`INVALID_IMAGE`, `INVALID_VIDEO`, `MODEL_UNAVAILABLE`, `CUDA_OUT_OF_MEMORY`,
`OUTPUT_LIMIT_EXCEEDED`, `FFMPEG_FAILED`, `STORAGE_FULL`, or `PROCESSING_FAILED`.

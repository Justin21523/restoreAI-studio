# RestorAI Studio — 3 minute portfolio demo

## 0:00–0:25 — Product problem

Open the public Workspace. Explain that restoration tools often stop at notebooks;
RestorAI adds durable Jobs, batch processing, provenance, expiry, and GPU back-pressure.
Point out the `DEMO` badge: Pages serves real precomputed results, not a fake hosted GPU.

## 0:25–1:00 — Real image quality

Select Archive portrait. Drag the Before/After control across the face, show the
single detected face, CodeFormer and Real-ESRGAN checksums, dimensions, and measured
time. Switch to Product detail and zoom into metal, leather, and wood texture.

## 1:00–1:30 — Video

Select City motion. Start both synchronized players and compare 24 versus 48 FPS.
Show that the three-second output retains audio and identify RIFE v4.25.

## 1:30–2:15 — Local real workflow

Switch to local Real Mode. Queue two images, open Batch Detail, and show serialized
GPU work. Open a Job Detail page to show live SSE events, input/output comparison,
parameters, hashes, model snapshot, expiry, download, cancel, and retry.

## 2:15–2:40 — Operations

Open System. Show PostgreSQL and Redis readiness, RQ queue depth, worker state,
RTX 5080 VRAM, storage capacity, and seven verified models under `/mnt/c/ai_models`.

## 2:40–3:00 — Engineering close

Show the architecture diagram. Explain the PostgreSQL outbox, one-GPU-owner policy,
RIFE-before-upscale decision, 24-hour file retention, metadata preservation, and
separation between a safe public Demo and the local CUDA environment.

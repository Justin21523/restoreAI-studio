# RestorAI Studio — 3 minute portfolio demo

## 0:00–0:25 — Immediate proof

Open Showcase. Point out Real-ESRGAN, GFPGAN, CodeFormer, and RIFE, the RTX 5080
evidence card, seven verified model files, and the explicit recorded-GPU label.

## 0:25–1:00 — Face restoration choices

Open Face Lab. Use the output tabs and Before/After slider to compare GFPGAN,
CodeFormer, and CodeFormer + Real-ESRGAN 2× on the same input. Explain fidelity,
strength, detected-face count, peak VRAM, full model SHA, and Artifact SHA.

## 1:00–1:30 — Motion and combined video

Open 24 → 48 FPS Motion and use 0.25× playback to make interpolation visible.
Then show RIFE → Real-ESRGAN: FPS and resolution both double while audio remains.

## 1:30–2:10 — Product-shaped workflow

Run the recorded pipeline and open Job Detail. Show named stages, durations,
parameters, model snapshot, input/output hashes and download. Open Jobs, select the
partial-failure Batch, retry the failed item, and show `retry_of_batch_id`.

## 2:10–2:40 — Real local operations

Open Models and System. Explain that Pages is static while the snapshot records the
real PostgreSQL, Redis/RQ, RTX 5080 worker, local Artifact storage and seven models
under `/mnt/c/ai_models`. In local Real mode, submit one actual file if time permits.

## 2:40–3:00 — Engineering close

Return to Showcase architecture. Explain transactional outbox recovery, one GPU
owner, RIFE-before-upscale, 24-hour file retention, persistent metadata, and why the
public Demo uses transparent evidence replay instead of an unsafe open GPU endpoint.

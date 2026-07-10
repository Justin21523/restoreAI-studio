# Demo source assets

The archive portrait and vintage camera masters were generated specifically for
RestorAI Studio with the built-in OpenAI image generation tool. They depict no
real person, product, or trademark. The original generated files are retained in
`docs/demo/sources/`.

## Archive portrait master

- Use: side-by-side GFPGAN and CodeFormer restoration plus CodeFormer/Real-ESRGAN 2×.
- Prompt intent: a realistic mid-century Taiwanese archive portrait with one
  unobstructed adult face, natural skin texture, period clothing, soft window
  light, no text, logos, borders, damage, or watermark.

## Product detail master

- Use: Real-ESRGAN 4×.
- Prompt intent: a realistic unbranded vintage mechanical camera on dark walnut,
  rich metal/leather/wood micro-texture, three-quarter close-up, no readable
  labels, logos, hands, or watermark.

## City motion

The short city scene is generated deterministically by
`scripts/build_demo_scenarios.py`. It contains a synthetic skyline, moving car,
rain, road motion, and a generated audio tone. No external media is used.

The same script creates degraded inputs and real GPU outputs. It records full model
and Artifact checksums, dimensions, FPS, audio preservation, execution stages, GPU
environment, cold-run timing, warm-run medians, and peak CUDA allocation in the
scenario and evidence manifests.

## Combined video

The combined case is derived deterministically from the project-owned city scene.
RIFE doubles 12 FPS to 24 FPS before Real-ESRGAN x2plus enlarges every source and
generated frame from 192×108 to 384×216. FFmpeg reattaches the source audio stream.

Public Demo interactions are transparent workflow replays. GitHub Pages never
claims to run inference; the checked-in outputs and evidence were generated locally
on the RTX 5080 recorded in `web/src/demo-evidence.json`.

`docs/demo/real-gpu-flow.webm` is a separate browser recording of the real mode. It
submits the project-owned product input through FastAPI, PostgreSQL/outbox, Redis/RQ
and a CUDA worker, waits for a succeeded Artifact, compares the result, then opens
the live Jobs and System pages. `scripts/capture_real_demo.py` reproduces the capture
against a running stack.

# Model registry

`MODEL_ROOT=/mnt/c/ai_models` is the only permitted weight root. Paths are resolved
from `restorai/model_manifest.yaml`, checked for root escape, size, readability,
and—in full verification mode—SHA-256.

## Required layout

```text
/mnt/c/ai_models/
  vision/upscale/RealESRGAN_x2plus.pth
  vision/upscale/RealESRGAN_x4plus.pth
  vision/face_restore/GFPGAN/GFPGANv1.4.pth
  vision/face_restore/Codeformer/codeformer.pth
  vision/face_restore/facelib/detection_Resnet50_Final.pth
  vision/face_restore/facelib/parsing_parsenet.pth
  vision/flow/rife_v4.25/flownet.pkl
```

Run `restorai models verify` after installation and before each release. The worker
also performs full verification once before accepting work; forked Jobs then use a
fast size/readability check to avoid re-hashing gigabytes on every request.

`MODEL_AUTO_DOWNLOAD=false` is deliberate. The only download command is the
explicit operator action `restorai models install-missing`, which installs manifest
entries to temporary files and validates their size/hash before atomic replacement.

## Licensing

- Real-ESRGAN: BSD-3-Clause upstream; verify model-specific terms.
- GFPGAN: Apache-2.0 plus upstream third-party notices.
- CodeFormer: NTU S-Lab License 1.0; review before hosted/commercial use.
- Practical-RIFE: MIT upstream.
- facexlib: MIT upstream.

Weights are not committed to this repository. See `THIRD_PARTY_NOTICES.md`.

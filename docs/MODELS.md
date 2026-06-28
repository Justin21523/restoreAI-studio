# Models

RestorAI Studio is designed around a centralized AI Warehouse so model weights can be shared across projects.

| Model family | Intended use | Prototype status |
| --- | --- | --- |
| Real-ESRGAN | Image super-resolution | Adapter scaffold exists; true inference needs weights and dependency pinning. |
| GFPGAN / CodeFormer | Face restoration | Planned/scaffolded. |
| RIFE | Video frame interpolation | Planned/scaffolded. |
| EDVR | Video super-resolution | Listed in downloader config. |

The public GitHub Pages demo intentionally does not load these models. It uses deterministic Canvas processing so interviewers can evaluate the workflow without GPU, private services, or multi-hundred-MB model downloads.

For real model work, first harden dependency pins, download weights into the warehouse, and add model-specific integration tests.

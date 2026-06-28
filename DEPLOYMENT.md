# Deployment

Primary deployment is GitHub Pages:

- Demo URL: `https://justin21523.github.io/restoreAI-studio/`
- Workflow: `.github/workflows/pages.yml`
- Artifact: `dist/`, generated from `portfolio-web/` and pushed to the legacy `gh-pages` branch

The Pages workflow runs Python compile checks, pytest smoke tests, the FastAPI demo smoke script, and the static demo build before publishing.

```bash
python -m compileall -q api core utils scripts ui webapp
python -m pytest -q
python scripts/smoke_demo.py
python scripts/build_static_demo.py
```

Docker demo files are retained for self-hosted deployments that need an Nginx static frontend plus the `webapp` FastAPI demo backend.

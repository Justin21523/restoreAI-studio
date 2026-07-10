from __future__ import annotations

import json
from pathlib import Path

import typer

from restorai.model_registry import ModelRegistry

app = typer.Typer(help="RestorAI Studio administration and processing CLI")
models_app = typer.Typer(help="Verify and install model weights")
app.add_typer(models_app, name="models")


@models_app.command("verify")
def verify_models(fast: bool = typer.Option(False, help="Skip SHA-256 calculation")) -> None:
    registry = ModelRegistry()
    statuses = registry.verify_all(verify_hash=not fast)
    typer.echo(
        json.dumps(
            [
                {
                    "model_id": item.model_id,
                    "path": str(item.path),
                    "available": item.available,
                    "valid": item.valid,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                    "error": item.error,
                }
                for item in statuses
            ],
            indent=2,
        )
    )
    if any(not item.valid for item in statuses):
        raise typer.Exit(code=1)


@models_app.command("install-missing")
def install_missing() -> None:
    registry = ModelRegistry()
    installed = registry.install_missing()
    typer.echo(json.dumps({"installed": [str(path) for path in installed]}, indent=2))


@app.command("process-directory")
def process_directory(
    directory: Path,
    operation: str = typer.Option("upscale"),
    pattern: str = typer.Option("*"),
) -> None:
    """Submit a directory as a batch through the running local API."""
    from restorai.services.client import submit_directory

    result = submit_directory(directory, operation=operation, pattern=pattern)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()

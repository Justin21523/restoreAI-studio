#!/usr/bin/env python3
"""
Generate support bundle for debugging and bug reports
"""
import json
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import shutil
from typing import Dict, Any

from api.config import get_settings
from utils.artifacts_index import artifact_index
from utils.metrics import metrics_collector

settings = get_settings()


def generate_support_bundle(
    output_path: Path,
    include_logs: bool = True,
    include_artifacts: bool = False,
    max_size_mb: int = 20,
):
    """Generate support bundle for debugging"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        bundle_info = {
            "generated_at": datetime.now().isoformat(),
            "bundle_version": "1.0",
            "components_included": [],
        }

        # 1. System configuration
        config_data = {
            "settings": {
                k: v
                for k, v in settings.dict().items()
                if not k.lower().endswith(("key", "secret", "password"))
            },
            "paths": {k: str(v) for k, v in settings.paths.items()},
        }

        config_file = temp_path / "configuration.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)
        bundle_info["components_included"].append("configuration")

        # 2. Recent logs
        if include_logs:
            log_files = list(Path("logs").glob("*.log"))
            for log_file in log_files[:5]:  # Limit to 5 most recent logs
                if log_file.stat().st_size < 10 * 1024 * 1024:  # 10MB limit per file
                    shutil.copy2(log_file, temp_path / log_file.name)
                    bundle_info["components_included"].append(f"logs:{log_file.name}")

        # 3. Recent artifacts summary
        artifacts_summary = artifact_index.get_artifact_stats()
        artifacts_file = temp_path / "artifacts_summary.json"
        with open(artifacts_file, "w") as f:
            json.dump(artifacts_summary, f, indent=2)
        bundle_info["components_included"].append("artifacts_summary")

        # 4. Metrics snapshot
        metrics_file = temp_path / "metrics.txt"
        with open(metrics_file, "w") as f:
            f.write(metrics_collector.get_metrics_text())
        bundle_info["components_included"].append("metrics")

        # 5. Recent errors
        errors_file = temp_path / "recent_errors.json"
        # This would collect recent errors from logs - simplified for now
        recent_errors = {"note": "Error collection would be implemented here"}
        with open(errors_file, "w") as f:
            json.dump(recent_errors, f, indent=2)
        bundle_info["components_included"].append("recent_errors")

        # 6. Bundle info
        info_file = temp_path / "bundle_info.json"
        with open(info_file, "w") as f:
            json.dump(bundle_info, f, indent=2)

        # Create zip file
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_path.iterdir():
                zipf.write(file_path, file_path.name)

        # Check size
        bundle_size = output_path.stat().st_size / (1024 * 1024)
        if bundle_size > max_size_mb:
            print(
                f"Warning: Bundle size {bundle_size:.1f}MB exceeds {max_size_mb}MB limit"
            )

        return bundle_info


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate RestorAI support bundle")
    parser.add_argument(
        "--output",
        "-o",
        default="restorai_support_bundle.zip",
        help="Output bundle file path",
    )
    parser.add_argument(
        "--no-logs", action="store_true", help="Exclude log files from bundle"
    )
    parser.add_argument(
        "--max-size", type=int, default=20, help="Maximum bundle size in MB"
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    print("Generating RestorAI support bundle...")

    try:
        bundle_info = generate_support_bundle(
            output_path=output_path,
            include_logs=not args.no_logs,
            max_size_mb=args.max_size,
        )

        print(f"Support bundle generated: {output_path}")
        print(f"Bundle size: {output_path.stat().st_size / (1024 * 1024):.1f}MB")
        print(f"Components included: {', '.join(bundle_info['components_included'])}")

    except Exception as e:
        print(f"Failed to generate support bundle: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

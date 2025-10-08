#!/usr/bin/env python3
"""
AI Warehouse Validation Script
Validates warehouse setup and provides diagnostic information.
"""

import sys
from pathlib import Path
import os
import json
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import get_settings
from core.esrgan import ESRGANProcessor


class WarehouseValidator:
    """Validates AI warehouse setup and configuration"""

    def __init__(self):
        self.settings = get_settings()
        self.issues = []
        self.warnings = []
        self.info = []

    def validate_paths(self) -> bool:
        """Validate all warehouse paths exist and are accessible"""
        print("🔍 Validating warehouse paths...")

        paths_ok = True

        for name, path in self.settings.PATHS.items():
            if not path.exists():
                self.issues.append(f"Missing path: {name} -> {path}")
                paths_ok = False
            elif not os.access(path, os.R_OK):
                self.issues.append(f"No read access: {name} -> {path}")
                paths_ok = False
            elif not os.access(path, os.W_OK):
                self.warnings.append(f"No write access: {name} -> {path}")
            else:
                self.info.append(f"✅ {name}: {path}")

        return paths_ok

    def validate_models(self) -> Dict[str, Any]:
        """Validate available models in warehouse"""
        print("🔍 Validating models...")

        models_dir = self.settings.PATHS["models"]
        available_models = []
        missing_models = []

        # Expected models
        expected_models = [
            "RealESRGAN_x4plus.pth",
            "RealESRGAN_x4plus_anime_6B.pth",
            "realesr-animevideov3-x4plus.pth",
        ]

        for model_name in expected_models:
            model_path = models_dir / model_name
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                available_models.append(
                    {
                        "name": model_name,
                        "path": str(model_path),
                        "size_mb": round(size_mb, 2),
                    }
                )
                self.info.append(f"✅ Model: {model_name} ({size_mb:.1f}MB)")
            else:
                missing_models.append(model_name)
                self.warnings.append(f"⚠️  Missing model: {model_name}")

        # Scan for other models
        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                if model_file.name not in expected_models:
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    available_models.append(
                        {
                            "name": model_file.name,
                            "path": str(model_file),
                            "size_mb": round(size_mb, 2),
                        }
                    )
                    self.info.append(
                        f"📦 Extra model: {model_file.name} ({size_mb:.1f}MB)"
                    )

        return {
            "available": available_models,
            "missing": missing_models,
            "total_count": len(available_models),
        }

    def validate_esrgan_processor(self) -> Dict[str, Any]:
        """Test ESRGAN processor initialization"""
        print("🔍 Testing ESRGAN processor...")

        try:
            processor = ESRGANProcessor()

            # Test basic properties
            result = {
                "device": str(processor.device),
                "model_path": (
                    str(processor.model_path)
                    if hasattr(processor, "model_path")
                    else None
                ),
                "initialized": False,
                "error": None,
            }

            # Test model loading (if model exists)
            if hasattr(processor, "model_path") and processor.model_path.exists():
                try:
                    processor.load_model()
                    result["initialized"] = True
                    self.info.append("✅ ESRGAN processor loaded successfully")
                except Exception as e:
                    result["error"] = str(e)
                    self.warnings.append(f"⚠️  ESRGAN load error: {e}")
            else:
                self.warnings.append("⚠️  ESRGAN model not found - download required")

            return result

        except Exception as e:
            self.issues.append(f"ESRGAN processor error: {e}")
            return {"error": str(e), "initialized": False}

    def check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies"""
        print("🔍 Checking dependencies...")

        required_packages = [
            "torch",
            "torchvision",
            "opencv-python",
            "fastapi",
            "gradio",
            "Pillow",
            "numpy",
        ]

        installed = []
        missing = []

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                installed.append(package)
                self.info.append(f"✅ Package: {package}")
            except ImportError:
                missing.append(package)
                self.issues.append(f"Missing package: {package}")

        return {
            "installed": installed,
            "missing": missing,
            "total_required": len(required_packages),
        }

    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space in warehouse"""
        print("🔍 Checking disk space...")

        disk_info = {}

        for name, path in self.settings.PATHS.items():
            try:
                stat = os.statvfs(path)
                free_bytes = stat.f_bavail * stat.f_frsize
                total_bytes = stat.f_blocks * stat.f_frsize
                used_bytes = total_bytes - free_bytes

                free_gb = free_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
                used_percent = (used_bytes / total_bytes) * 100

                disk_info[name] = {
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "used_percent": round(used_percent, 1),
                }

                if free_gb < 5:
                    self.warnings.append(
                        f"⚠️  Low disk space in {name}: {free_gb:.1f}GB free"
                    )
                else:
                    self.info.append(f"✅ Disk space {name}: {free_gb:.1f}GB free")

            except Exception as e:
                self.warnings.append(f"Could not check disk space for {name}: {e}")

        return disk_info

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        print("\n" + "=" * 60)
        print("🏭 AI WAREHOUSE VALIDATION REPORT")
        print("=" * 60)

        # Run all validations
        paths_valid = self.validate_paths()
        models_info = self.validate_models()
        esrgan_info = self.validate_esrgan_processor()
        deps_info = self.check_dependencies()
        disk_info = self.check_disk_space()

        # Generate report
        report = {
            "timestamp": self.settings.timestamp,
            "warehouse_root": str(self.settings.PATHS["warehouse_root"]),
            "validation": {
                "paths_valid": paths_valid,
                "models": models_info,
                "esrgan": esrgan_info,
                "dependencies": deps_info,
                "disk_space": disk_info,
            },
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info,
            "overall_status": "OK" if not self.issues else "ISSUES_FOUND",
        }

        # Print summary
        print(f"\n📊 SUMMARY:")
        print(f"Warehouse Root: {report['warehouse_root']}")
        print(f"Models Available: {models_info['total_count']}")
        print(
            f"Dependencies: {len(deps_info['installed'])}/{deps_info['total_required']}"
        )
        print(f"Overall Status: {report['overall_status']}")

        if self.issues:
            print(f"\n❌ ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  • {issue}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")

        if self.info:
            print(f"\n✅ STATUS ({len(self.info)}):")
            for info in self.info:
                print(f"  • {info}")

        return report


def main():
    """Main validation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate AI warehouse setup")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--save", help="Save report to file")
    parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix common issues"
    )

    args = parser.parse_args()

    try:
        validator = WarehouseValidator()
        report = validator.generate_report()

        if args.json:
            print("\n" + json.dumps(report, indent=2))

        if args.save:
            with open(args.save, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Report saved to: {args.save}")

        if args.fix:
            print("\n🔧 Attempting to fix issues...")
            fix_common_issues(validator.settings)

        # Return exit code
        return 0 if report["overall_status"] == "OK" else 1

    except Exception as e:
        print(f"\n💥 Validation failed: {e}")
        return 2


def fix_common_issues(settings):
    """Attempt to fix common warehouse issues"""
    print("Creating missing directories...")

    for name, path in settings.PATHS.items():
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created: {name} -> {path}")
            except Exception as e:
                print(f"❌ Failed to create {name}: {e}")

    print("🔧 Basic fixes applied. Re-run validation to check results.")


if __name__ == "__main__":
    sys.exit(main())

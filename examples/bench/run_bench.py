#!/usr/bin/env python3
"""
Micro-benchmark for RestorAI performance testing
"""
import time
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import cv2

from core.esrgan import ESRGAN
from core.gfpgan import GFPGAN
from core.registry import model_registry
from api.config import get_settings

settings = get_settings()


class Benchmark:
    """Performance benchmark for RestorAI models"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paths["outputs"] / "bench"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def create_test_images(self) -> Dict[str, np.ndarray]:
        """Create test images of different sizes"""
        test_images = {
            "small": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "medium": np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            "large": np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8),
        }
        return test_images

    def benchmark_esrgan(self, repeats: int = 3) -> List[Dict[str, Any]]:
        """Benchmark ESRGAN models"""
        results = []
        test_images = self.create_test_images()

        esrgan_models = [
            ("realesrgan-x2plus", 2),
            ("realesrgan-x4plus", 4),
        ]

        for model_name, scale in esrgan_models:
            # Check if model is available
            is_valid, message = model_registry.validate_model(model_name)
            if not is_valid:
                print(f"Skipping {model_name}: {message}")
                continue

            print(f"Benchmarking {model_name}...")
            model = ESRGAN(model_name=model_name, device=settings.device)

            for size_name, test_image in test_images.items():
                print(f"  Testing {size_name} image...")

                # Warmup
                _ = model.infer(test_image)

                # Benchmark
                times = []
                vram_usage = []

                for i in range(repeats):
                    start_time = time.time()
                    result = model.infer(test_image)
                    end_time = time.time()

                    times.append((end_time - start_time) * 1000)  # Convert to ms

                    # Get VRAM usage if available
                    perf_stats = model.get_performance_stats()
                    vram_usage.append(perf_stats.get("vram_used_gb", 0))

                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_vram = np.mean(vram_usage) if vram_usage else 0

                results.append(
                    {
                        "model": model_name,
                        "test_case": f"esrgan_{size_name}",
                        "image_size": f"{test_image.shape[1]}x{test_image.shape[0]}",
                        "scale": scale,
                        "avg_latency_ms": avg_time,
                        "std_latency_ms": std_time,
                        "avg_vram_gb": avg_vram,
                        "repeats": repeats,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return results

    def benchmark_gfpgan(self, repeats: int = 3) -> List[Dict[str, Any]]:
        """Benchmark GFPGAN models"""
        results = []
        test_images = self.create_test_images()

        face_models = ["gfpgan", "codeformer"]

        for model_name in face_models:
            # Check if model is available
            is_valid, message = model_registry.validate_model(model_name)
            if not is_valid:
                print(f"Skipping {model_name}: {message}")
                continue

            print(f"Benchmarking {model_name}...")
            model = GFPGAN(model_name=model_name, device=settings.device)

            for size_name, test_image in test_images.items():
                print(f"  Testing {size_name} image...")

                # Warmup
                _ = model.infer(test_image)

                # Benchmark
                times = []
                vram_usage = []

                for i in range(repeats):
                    start_time = time.time()
                    result = model.infer(test_image, strength=0.5)
                    end_time = time.time()

                    times.append((end_time - start_time) * 1000)

                    perf_stats = model.get_performance_stats()
                    vram_usage.append(perf_stats.get("vram_used_gb", 0))

                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_vram = np.mean(vram_usage) if vram_usage else 0

                results.append(
                    {
                        "model": model_name,
                        "test_case": f"face_{size_name}",
                        "image_size": f"{test_image.shape[1]}x{test_image.shape[0]}",
                        "avg_latency_ms": avg_time,
                        "std_latency_ms": std_time,
                        "avg_vram_gb": avg_vram,
                        "repeats": repeats,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return results

    def run_benchmark_suite(self, repeats: int = 3) -> Path:
        """Run complete benchmark suite"""
        print("Starting RestorAI Benchmark Suite")
        print("=" * 50)

        # Run benchmarks
        all_results = []
        all_results.extend(self.benchmark_esrgan(repeats))
        all_results.extend(self.benchmark_gfpgan(repeats))

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"benchmark_{timestamp}.csv"
        json_path = self.output_dir / f"benchmark_{timestamp}.json"

        # Save as CSV
        if all_results:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)

        # Save as JSON
        with open(json_path, "w") as f:
            json.dump(
                {
                    "benchmark_info": {
                        "timestamp": timestamp,
                        "repeats": repeats,
                        "device": settings.device,
                        "total_tests": len(all_results),
                    },
                    "results": all_results,
                },
                f,
                indent=2,
            )

        # Print summary
        self._print_summary(all_results)

        print(f"\nResults saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")

        return csv_path

    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print benchmark summary"""
        print("\nBenchmark Summary")
        print("=" * 50)

        for result in results:
            print(
                f"{result['test_case']:20} | "
                f"{result['avg_latency_ms']:6.1f} ms ± {result['std_latency_ms']:4.1f} | "
                f"{result['avg_vram_gb']:4.1f} GB | "
                f"{result['model']}"
            )


def main():
    """Main benchmark runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Run RestorAI benchmark suite")
    parser.add_argument(
        "--repeats", "-r", type=int, default=3, help="Number of repetitions per test"
    )
    parser.add_argument("--output-dir", "-o", help="Output directory for results")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    benchmark = Benchmark(output_dir)

    try:
        results_file = benchmark.run_benchmark_suite(repeats=args.repeats)
        print(f"\nBenchmark completed successfully!")
        return 0
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

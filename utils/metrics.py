import time
from typing import Dict, Any, List
from dataclasses import dataclass
from collections import defaultdict
import threading
import psutil
import torch

from api.config import get_settings
from utils.logging import get_logger

settings = get_settings()
logger = get_logger("metrics")


@dataclass
class MetricSample:
    """Single metric sample"""

    value: float
    timestamp: float
    labels: Dict[str, str]


class MetricsCollector:
    """Collect and expose application metrics"""

    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)
        self.lock = threading.Lock()

        # Initialize metrics
        self._init_metrics()

        # Start background collection
        self._start_background_collection()

    def _init_metrics(self):
        """Initialize default metrics"""
        # Request counters
        self.counters["requests_total"] = 0
        self.counters["requests_by_status"] = 0
        self.counters["requests_by_task"] = 0

        # Processing metrics
        self.gauges["processing_duration_seconds"] = {}
        self.gauges["vram_usage_bytes"] = {}
        self.gauges["active_models"] = 0

        # System metrics
        self.gauges["system_cpu_percent"] = 0
        self.gauges["system_memory_bytes"] = 0
        self.gauges["system_disk_usage_bytes"] = 0

    def _start_background_collection(self):
        """Start background system metrics collection"""

        def collect_system_metrics():
            while True:
                try:
                    self._update_system_metrics()
                    time.sleep(30)  # Collect every 30 seconds
                except Exception as e:
                    logger.error(f"System metrics collection failed: {e}")
                    time.sleep(60)

        thread = threading.Thread(
            target=collect_system_metrics, name="metrics_collector", daemon=True
        )
        thread.start()

    def _update_system_metrics(self):
        """Update system-level metrics"""
        try:
            # CPU usage
            self.gauges["system_cpu_percent"] = psutil.cpu_percent()

            # Memory usage
            memory = psutil.virtual_memory()
            self.gauges["system_memory_bytes"] = memory.used

            # Disk usage
            disk = psutil.disk_usage("/")
            self.gauges["system_disk_usage_bytes"] = disk.used

            # GPU memory if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i)
                    self.gauges[f"gpu_{i}_memory_bytes"] = allocated

        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")

    def increment_counter(
        self, name: str, labels: Dict[str, str] = None, value: int = 1
    ):
        """Increment a counter metric"""
        with self.lock:
            key = self._get_metric_key(name, labels)
            self.counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        with self.lock:
            key = self._get_metric_key(name, labels)
            self.gauges[key] = value

    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a histogram value"""
        with self.lock:
            key = self._get_metric_key(name, labels)
            self.histograms[key].append(value)

            # Keep only recent samples (last 1000)
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]

    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create metric key with labels"""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        lines = []

        with self.lock:
            # Counters
            for key, value in self.counters.items():
                lines.append(f"restorai_{key} {value}")

            # Gauges
            for key, value in self.gauges.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        lines.append(f"restorai_{key}{{{subkey}}} {subvalue}")
                else:
                    lines.append(f"restorai_{key} {value}")

            # Histograms (as summaries for simplicity)
            for key, values in self.histograms.items():
                if values:
                    count = len(values)
                    total = sum(values)
                    lines.append(f"restorai_{key}_count {count}")
                    lines.append(f"restorai_{key}_sum {total}")

        return "\n".join(lines) + "\n"

    def record_request(self, method: str, path: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        labels = {"method": method, "path": path, "status": str(status_code)}

        self.increment_counter("requests_total", labels)
        self.observe_histogram("request_duration_seconds", duration, labels)

    def record_processing(
        self, task_type: str, model: str, duration: float, success: bool
    ):
        """Record processing metrics"""
        labels = {"task": task_type, "model": model, "success": str(success).lower()}

        self.increment_counter("processing_total", labels)
        self.observe_histogram("processing_duration_seconds", duration, labels)


# Global metrics collector
metrics_collector = MetricsCollector()

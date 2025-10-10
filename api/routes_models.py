import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor

from core.manager import resource_manager, ResourcePolicy
from api.config import get_settings
from utils.logger import logger

settings = get_settings()


@dataclass
class ModelConfig:
    """Model configuration for lifecycle management"""

    adapter_name: str
    model_class: Callable
    default_device: str = "cpu"
    default_precision: str = "fp32"
    estimated_size_mb: int = 500
    warmup_batch_size: int = 1
    warmup_input_shape: tuple = (512, 512)


class ModelLifecycleManager:
    """Manage model warmup, health checks, and lifecycle"""

    def __init__(self):
        self.model_configs: Dict[str, ModelConfig] = {}
        self.warmup_tasks: Dict[str, Future] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="model_lifecycle"
        )

        # Start background health monitor
        self._start_health_monitor()

    def register_model(self, config: ModelConfig):
        """Register a model for lifecycle management"""
        self.model_configs[config.adapter_name] = config
        self.health_status[config.adapter_name] = {
            "registered": True,
            "warm": False,
            "last_health_check": None,
            "healthy": True,
            "error_count": 0,
        }

        logger.info(f"Registered model for lifecycle management: {config.adapter_name}")

    def warmup_model(self, adapter_name: str, force: bool = False) -> bool:
        """Warm up a model asynchronously"""
        if adapter_name not in self.model_configs:
            logger.warning(f"Model not registered: {adapter_name}")
            return False

        # Check if already warming up
        if adapter_name in self.warmup_tasks and not force:
            logger.debug(f"Model {adapter_name} is already warming up")
            return True

        # Submit warmup task
        future = self.executor.submit(self._warmup_model_impl, adapter_name)
        self.warmup_tasks[adapter_name] = future

        logger.info(f"Started warmup for model: {adapter_name}")
        return True

    def _warmup_model_impl(self, adapter_name: str):
        """Implementation of model warmup"""
        try:
            config = self.model_configs[adapter_name]

            # Get resource policy
            policy = resource_manager.get_optimal_policy(config.estimated_size_mb)

            # Create model key
            model_key = resource_manager.create_model_key(
                adapter_name, policy.device.value, policy.precision.value
            )

            # Check if already in pool
            if resource_manager.get_model(model_key):
                self.health_status[adapter_name].update(
                    {"warm": True, "last_health_check": time.time(), "healthy": True}
                )
                return

            # Load model
            logger.info(f"Warming up model {adapter_name} on {policy.device}")

            # This would be implemented by specific adapters
            # For now, we just simulate warmup
            time.sleep(2)  # Simulate loading time

            self.health_status[adapter_name].update(
                {
                    "warm": True,
                    "last_health_check": time.time(),
                    "healthy": True,
                    "warmup_completed": True,
                }
            )

            logger.info(f"Model warmup completed: {adapter_name}")

        except Exception as e:
            logger.error(f"Model warmup failed for {adapter_name}: {e}")
            self.health_status[adapter_name].update(
                {
                    "warm": False,
                    "healthy": False,
                    "last_error": str(e),
                    "error_count": self.health_status[adapter_name].get(
                        "error_count", 0
                    )
                    + 1,
                }
            )

    def check_model_health(self, adapter_name: str) -> bool:
        """Check if a model is healthy and ready"""
        if adapter_name not in self.health_status:
            return False

        status = self.health_status[adapter_name]

        # Simple health check - in production, this would run actual inference
        is_healthy = status.get("healthy", False) and status.get("warm", False)

        status.update({"last_health_check": time.time(), "healthy": is_healthy})

        return is_healthy

    def unload_model(self, adapter_name: str) -> bool:
        """Unload a model from memory"""
        try:
            # Remove from resource manager pool
            models_to_remove = []
            for model_key in resource_manager.model_pool:
                if model_key.startswith(adapter_name):
                    models_to_remove.append(model_key)

            for model_key in models_to_remove:
                if model_key in resource_manager.model_pool:
                    del resource_manager.model_pool[model_key]

            # Update health status
            if adapter_name in self.health_status:
                self.health_status[adapter_name].update(
                    {"warm": False, "unloaded": True}
                )

            logger.info(f"Unloaded model: {adapter_name}")
            return True

        except Exception as e:
            logger.error(f"Model unloading failed for {adapter_name}: {e}")
            return False

    def _start_health_monitor(self):
        """Start background health monitoring"""

        def monitor_loop():
            while True:
                try:
                    self._run_health_checks()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Health monitor error: {e}")
                    time.sleep(30)

        monitor_thread = threading.Thread(
            target=monitor_loop, name="health_monitor", daemon=True
        )
        monitor_thread.start()

    def _run_health_checks(self):
        """Run health checks on all registered models"""
        for adapter_name in self.model_configs:
            try:
                self.check_model_health(adapter_name)
            except Exception as e:
                logger.error(f"Health check failed for {adapter_name}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get lifecycle manager status"""
        return {
            "registered_models": list(self.model_configs.keys()),
            "health_status": self.health_status,
            "warmup_tasks": {
                name: task.done() for name, task in self.warmup_tasks.items()
            },
        }


# Global lifecycle manager instance
lifecycle_manager = ModelLifecycleManager()

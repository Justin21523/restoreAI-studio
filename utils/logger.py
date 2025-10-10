import logging
import sys
from api.config import get_settings

settings = get_settings()


def setup_logger(name: str = "restorai"):
    """Setup application logger"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, settings.log_level.upper()))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


# Global logger instance
logger = setup_logger()

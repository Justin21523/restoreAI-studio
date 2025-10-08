# api/config.py
import pydantic
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    AI_WAREHOUSE_ROOT: str = "~/ai-warehouse"
    MODELS_DIR: str = "${AI_WAREHOUSE_ROOT}/models"
    DATASETS_DIR: str = "${AI_WAREHOUSE_ROOT}/datasets"
    CACHE_DIR: str = "${AI_WAREHOUSE_ROOT}/cache"
    INPUT_DIR: str = "./data/input"
    OUTPUT_DIR: str = "./data/output"
    TEMP_DIR: str = "./data/temp"

    class Config:
        env_file = ".env"
        case_sensitive = False

    def resolve(self, p: str) -> Path:
        return (
            Path(p.replace("${AI_WAREHOUSE_ROOT}", self.AI_WAREHOUSE_ROOT))
            .expanduser()
            .resolve()
        )

    @property
    def PATHS(self):
        return {
            "models": self.resolve(self.MODELS_DIR),
            "datasets": self.resolve(self.DATASETS_DIR),
            "cache": self.resolve(self.CACHE_DIR),
            "input": self.resolve(self.INPUT_DIR),
            "output": self.resolve(self.OUTPUT_DIR),
            "temp": self.resolve(self.TEMP_DIR),
        }


settings = Settings()

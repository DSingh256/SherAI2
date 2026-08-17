"""
VanRakshak AI - Backend Configuration
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings from environment variables"""

    # API Configuration
    APP_NAME: str = "VanRakshak AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./vanrakshak.db"
    SQLALCHEMY_ECHO: bool = False

    # Storage Configuration
    RAW_STORAGE_PATH: str = "storage/raw"
    PROCESSED_STORAGE_PATH: str = "storage/processed"
    QUARANTINE_STORAGE_PATH: str = "storage/quarantine"
    SEGMENTED_STORAGE_PATH: str = "storage/segmented"

    # Image Upload Configuration
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_IMAGE_EXTENSIONS: list = ["jpg", "jpeg", "png", "gif", "tiff"]

    # Confidence Thresholds (Configurable)
    HIGH_CONFIDENCE_THRESHOLD: float = 0.90
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.60
    LOW_CONFIDENCE_THRESHOLD: float = 0.00

    # Quality Gate Thresholds
    BLUR_THRESHOLD: float = 100.0  # Laplacian variance threshold
    MIN_BRIGHTNESS: int = 10  # Minimum average brightness (0-255)
    MAX_BRIGHTNESS: int = 245  # Maximum average brightness (0-255)

    # Model Configuration
    MEGADETECTOR_CONFIDENCE_THRESHOLD: float = 0.5
    SPECIESNET_CONFIDENCE_THRESHOLD: float = 0.5
    SEMANTIC_CONFIDENCE_THRESHOLD: float = 0.5

    # Feature Flags
    ENABLE_MEGADETECTOR: bool = True
    ENABLE_SPECIESNET: bool = True
    ENABLE_OPENCLIP: bool = True  # Phase 5
    ENABLE_SAM_SEGMENTATION: bool = True  # Phase 7

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

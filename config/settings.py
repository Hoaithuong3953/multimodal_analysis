"""Application settings loaded from .env file"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional, Literal

LOG_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables (.env)
    """
    # Logging configuration
    log_level: LOG_LEVEL = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)"
    )
    log_format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        description="Default log format (can be customized if needed)"
    )
    log_date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date/time format for logs"
    )
    log_to_file: bool = Field(
        default=False,
        description="If true, enable logging to a file"
    )
    log_file_path: str = Field(
        default="logs/app.log",
        description="Path to log file"
    )
    log_file_rotation: str = Field(
        default="midnight",
        description="Log file rotation interval"
    )
    log_file_retention: int = Field(
        default=7,
        ge=1,
        description="Number of days to retain log files"
    )

    # Model names
    text_model_name: str = Field(
        default="distilbert-base-multilingual-cased",
        description="Hugging Face model name for text encoder (DistilBERT multilingual for Vietnamese support)"
    )
    image_model_name: str = Field(
        default="resnet18",
        description="Torchvision model name for image encoder (resnet18, resnet50, etc.)"
    )
    checkpoint_path: Optional[str] = Field(
        default="config/checkpoint.pt",
        description="Path to fusion model checkpoint; if file exists, it is loaded at startup"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
"""
Logging configuration helpers and shared logger instance
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from config.settings import settings

def setup_logger(name: str = "multimodal-ai") -> logging.Logger:
    """
    Configure and return a named logger with console and optional file output

    Configuration is loaded from settings (LOG_LEVEL, LOG_TO_FILE, LOG_FILE_*)

    Args:
        name: Logger name (default 'multimodal-ai')

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        logger.setLevel(level)

        # Use custom format if provided
        default_format = "%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s"
        formatter = logging.Formatter(
            fmt=settings.log_format or default_format,
            datefmt=settings.log_date_format
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if settings.log_to_file:
            log_path = Path(settings.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handlers = logging.handlers.TimedRotatingFileHandler(
                filename=log_path,
                when=settings.log_file_rotation,
                interval=1,
                backupCount=settings.log_file_retention,
                encoding="utf-8"
            )

            file_handlers.setFormatter(formatter)
            logger.addHandler(file_handlers)

    logger.propagate = False
    return logger

logger = setup_logger()
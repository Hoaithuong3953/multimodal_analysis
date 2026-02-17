"""Custom exceptions for the multimodal analysis application"""

from typing import Any, Dict, Optional


class AppError(Exception):
    """
    Base exception for all application errors

    Attributes:
        message: Human-readable error message
        code: Short error code for identification
    """

    code: str = "GENERAL_ERROR"
    message: str = "An application error occurred"

    def __init__(
        self,
        code: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs: Any
    ):
        self.code = code or self.code
        self.message = message or self.message
        self.extra = kwargs
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for response"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                **self.extra,
            }
        }
    
    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

class InvalidInputError(AppError):
    """Raised when input data is invalid (empty text, invalid image, etc.)"""
    code = "INVALID_INPUT"
    message = "The provided input is invalid or malformed"

class AnalysisFailedError(AppError):
    """Raised when analysis pipeline or model inference fails"""
    code = "ANALYSIS_FAILED"
    message = "The analysis process failed. Please try again later"

class ModelNotLoadedError(AppError):
    """Raised when model fails to load or checkpoint is missing"""
    code = "MODEL_NOT_LOADED"
    message = "The AI model is not available or failed to initialize"
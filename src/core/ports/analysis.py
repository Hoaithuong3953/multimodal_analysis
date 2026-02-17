"""Ports (interface) for the analysis service"""
from typing import Protocol, Optional
from src.core import AnalysisResult

class IAnalysisService(Protocol):
    """Interface for the analysis service"""

    def analyze(self, text: str, image_path: Optional[str] = None) -> AnalysisResult:
        """
        Analyze input text and optional image

        Args:
            text (str): Customer message text
            image_path (Optional[str]): Path to attached image (if any)

        Returns:
            AnalysisResult: Analysis result with sentiment, intent and suggestions
        """
        ...
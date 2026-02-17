"""Gradio web interface for multimodal analysis"""
from __future__ import annotations

import uuid
from pathlib import Path

from typing import Optional
import gradio as gr

from config import logger
from src.analysis import get_analysis_service
from src.core import AppError

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "images"

def analyze_ui(text: str, image) -> dict:
    """
    Gradio handler for analysis requests
    
    Args:
        text: Customer message text input from UI
        image: Optional PIL Image object from UI

    Returns:
        Dictionary containing either:
            - AnalysisResult fields on success
            - Error dictionary with 'error' key on failure
    """
    logger.info("Received request")
    srv = get_analysis_service()
    image_path: Optional[str] = None

    try:
        if image is not None:
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            path = IMAGES_DIR / f"{uuid.uuid4().hex}.png"
            image.save(path, format="PNG")
            image_path = str(path)
        result = srv.analyze(text=text, image_path=image_path)
        return result.model_dump()
    except AppError as e:
        logger.warning(f"AppError: {e}")
        return e.to_dict()
    
demo = gr.Interface(
    fn=analyze_ui,
    inputs=[
        gr.Textbox(label="Customer message", lines=3),
        gr.Image(label="Optional image", type="pil")
    ],
    outputs=gr.JSON(label="Analysis result"),
    title="Multimodal CX Analyzer",
    description="Enter a message (and optional image to see sentiment, intent, repurchase and recommendations)"
)
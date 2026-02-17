"""Container for the analysis service with dependency injection"""
from functools import lru_cache

from config import settings
from src.analysis import (
    TextEncoder,
    ImageEncoder,
    MultimodalFusionModel,
    AnalysisService,
)

@lru_cache(maxsize=1)
def get_text_encoder() -> TextEncoder:
    """
    Get singleton TextEncoder instance

    Returns:
        TextEncoder instance configured with settings.text_model_name
        Subsequent calls return the same cached instance
    """
    return TextEncoder(model_name=settings.text_model_name)

@lru_cache(maxsize=1)
def get_image_encoder() -> ImageEncoder:
    """
    Get singleton ImageEncoder instance
    
    Returns:
        ImageEncoder instance configured with settings.image_model_name
        Subsequent calls return the same cached instance
    """
    return ImageEncoder(model_name=settings.image_model_name)

@lru_cache(maxsize=1)
def get_fusion_model() -> MultimodalFusionModel:
    """
    Get singleton MultimodalFusionModel instance

    Returns:
        MultimodalFusionModel instance with dimensions matching encoders
        Subsequent calls return the same cached instance
    """
    text_encoder = get_text_encoder()
    image_encoder = get_image_encoder()
    return MultimodalFusionModel(
        text_dim=text_encoder.hidden_size,
        image_dim=image_encoder.feature_dim,
    )

@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    """
    Get singleton AnalysisService instance with all dependencies wired

    Returns:
        AnalysisService instance with encoders and fusion model injected
        Subsequent calls return the same cached instance
    """
    text_encoder = get_text_encoder()
    image_encoder = get_image_encoder()
    fusion_model = get_fusion_model()
    return AnalysisService(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion_model=fusion_model,
        checkpoint_path=settings.checkpoint_path,
    )
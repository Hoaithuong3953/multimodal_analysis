"""Core data models for the multimodal analysis system"""
from pydantic import BaseModel, Field
from typing import Optional, Literal

SentimentType = Literal["positive", "neutral", "negative"]
IntentType = Literal["complaint", "additional_purchase", "inquiry"]

class AnalysisResult(BaseModel):
    """Result of the multimodal analysis containing sentiment, intent and recommendations"""
    # Core labels
    sentiment: SentimentType = Field(
        ...,
        description="Detect sentiment: 'positive', 'neutral' or 'negative'"
    )
    intent: IntentType = Field(
        ...,
        description="Customer intent: 'complaint', 'additional_purchase' or 'inquiry'"
    )
    will_rebuy: bool = Field(
        ...,
        description="Prediction if the customer will repurchase (True/False)"
    )

    # Confidence score (0-1)
    sentiment_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence for sentiment prediction"
    )
    will_rebuy_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence for will_rebuy prediction"
    )

    # Recommendations
    template_suggestion: str = Field(
        ...,
        description="Suggested reply template for the shop"
    )
    offer_suggestion: Optional[str] = Field(
        None,
        description="Suggested offer or discount (e.g., '10% off with code ABC')"
    )
    routing_suggestion: Optional[str] = Field(
        None,
        description="Suggested routing (e.g., 'Senior Customer Service' or 'Keep current')"
    )
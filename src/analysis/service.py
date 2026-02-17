"""Analysis service for multimodal customer experience analysis"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import logger
from src.core import (
    AppError,
    InvalidInputError,
    AnalysisFailedError,
    AnalysisResult,
)
from src.analysis.encoders import TextEncoder, ImageEncoder, DEVICE
from src.analysis.pipeline import MultimodalFusionModel

def _load_image(path: Optional[str]) -> torch.Tensor:
    """
    Load image from disk and return ImageNet-normalized tensor

    Args:
        path: Path to image file. If None, returns normalized zero tensor

    Returns:
        Image tensor of shape (1, 3, 244, 244), ImageNet-normalized
        (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    Raises:
        InvalidInputError: If image file cannot be opened
    """
    if path is None:
        zero_img = torch.zeros(1, 3, 244, 244, device=DEVICE)

        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        return (zero_img - mean) / std
    
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        raise InvalidInputError(
            message=f"Cannot open image at path: {path}"
        ) from e
    
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    return tfm(img).unsqueeze(0).to(DEVICE)

class AnalysisService:
    """
    Service for analyzing customer messages with multimodal inputs

    Orchestrates the full analysis pipeline:
    1. Encode text and image inputs
    2. Fuse embeddings via MultimodalFusionModel
    3. Predict sentiment and repurchase likelihood
    4. Derive intent and recommendations via rule-based logic

    Attributes:
        text_encoder: TextEncoder instance for text encoding
        image_encoder: ImageEncoder instance for image encoding
        model: MultimodalFusionModel instance for fusion and prediction
    """
    def __init__(
        self,
        text_encoder: TextEncoder,
        image_encoder: ImageEncoder,
        fusion_model: Optional[MultimodalFusionModel] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Initialize AnalysisService with encoders and fusion model.

        Args:
            text_encoder: TextEncoder instance
            image_encoder: ImageEncoder instance
            fusion_model: Optional MultimodalFusionModel
            checkpoint_path: Optional path to model checkpoint
        """
        logger.info("Initializing encoders and fusion model")
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        self.model = fusion_model or MultimodalFusionModel()
        self.model.to(DEVICE)
        self.model.eval()

        if checkpoint_path:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                try:
                    state = torch.load(ckpt, map_location=DEVICE)
                    self.model.load_state_dict(state)
                    logger.info(f"Loaded checkpoint from {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint path: {e!r}")
            else:
                logger.info(f"No checkpoint found at {ckpt}")

    @torch.no_grad()
    def analyze(self, text: str, image_path: Optional[str] = None) -> AnalysisResult:
        """
        Analyze customer message with optional image
        
        Args:
            text: Customer message text (required, non-empty)
            image_path: Optional path to image file. If None, uses zero image

        Returns:
            AnalysisResult containing:
                - sentiment: 'positive', 'neutral' or 'negative'
                - intent: 'complaint', 'additional_purchase' or 'inquiry'
                - will_rebuy: Boolean prediction
                - confidence scores for sentiment and repurchase
                - template_suggestion, offer_suggestion, routing_suggestion

        Raises:
            InvalidInputError: If text is empty or invalid
            AnalysisFailedError: If analysis pipeline fails unexpectedly
        """
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError(
                message="Text input is required and must be non-empty"
            )
        
        logger.info("Running analysis")

        try:
            text_emb = self.text_encoder(text).unsqueeze(0)
            image_tensor = _load_image(image_path)
            image_emb = self.image_encoder(image_tensor)

            sent_logits, rebuy_logits = self.model(text_emb, image_emb)

            sent_probs = F.softmax(sent_logits, dim=-1).squeeze(0)
            sent_idx = int(sent_probs.argmax().item())
            sent_conf = float(sent_probs.max().item())
            sentiment_label = ["negative", "neutral", "positive"][sent_idx]

            rebuy_prob = torch.sigmoid(rebuy_logits).squeeze(0).item()
            rebuy_conf = float(rebuy_prob)
            will_rebuy = bool(rebuy_prob >= 0.5)

            # Rule-based intent (from model sentiment + will_rebuy)
            if sentiment_label == "negative":
                intent = "complaint"
            elif sentiment_label == "positive" and will_rebuy:
                intent = "additional_purchase"
            else:
                intent = "inquiry"

            # Rule-based recommendations
            if sentiment_label == "negative":
                template = "Xin lỖi quý khách, chúng tôi sẽ kiểm tra và phản hồi sớm nhất"
                offer = "Mã giảm giá 10% cho đơn tiếp theo"
                routing = "Chuyển nhân viên CSKH cấp cao"
            elif sentiment_label == "positive" and will_rebuy:
                template = "Cảm ơn quý khác, chúng tôi rất vui vì quý khách hàng lòng"
                offer = "Gợi ý sản phẩm liên quan / combo"
                routing = "Giữ tại CSKH hiện tại"
            else:
                template = "Cảm ơn phản hồi của quý khách, chúng tôi sẽ hỗ trợ thêm nếu cần"
                offer = None
                routing = "Giữ CSKH hiện tại"

            result = AnalysisResult(
                sentiment=sentiment_label,
                intent=intent,
                will_rebuy=will_rebuy,
                sentiment_confidence=sent_conf,
                will_rebuy_confidence=rebuy_conf,
                template_suggestion=template,
                offer_suggestion=offer,
                routing_suggestion=routing,
            )
            return result
        except AppError:
            raise
        except Exception as e:
            logger.exception("Unexpected failure")
            raise AnalysisFailedError(
                message="Multimodal analysis failed unexpectedly"
            ) from e
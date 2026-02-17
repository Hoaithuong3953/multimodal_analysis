"""Text and image encoders for multimodal analysis"""
import torch
import torch.nn as nn
from torchvision import models as tv_models
from transformers import AutoTokenizer, AutoModel

from config import logger
from src.core import InvalidInputError, ModelNotLoadedError, AnalysisFailedError

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextEncoder(nn.Module):
    """
    Text encoder using HuggingFace model to load
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        logger.info(f"Loading model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        except Exception as e:
            logger.exception("Failed to load model")
            raise ModelNotLoadedError(
                message=f"Failed to load text encoder model '{self.model_name}'"
            ) from e
        
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.to(DEVICE)
        self.model.eval()
        self.hidden_size = getattr(self.model.config, "hidden_size", 768)
        logger.info("Model loaded, parameters frozen")

    @torch.no_grad()
    def forward(self, text: str) -> torch.Tensor:
        """
        Encode a single text string to normalized CLS embedding
        
        Args:
            text: Input text string (non-empty)

        Returns:
            Normalized embedding vector of shape (hidden_size,)
            Uses pooler_output if available, otherwise last_hidden_state[:, 0, :]

        Raises:
            InvalidInputError: If text is empty or not a string
            AnalysisFailedError: If encoding process fails
        """
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("Text input is required and must be non-empty")
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(DEVICE)

            outputs = self.model(**inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state[:, 0, :]
            
            emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb.squeeze(0)
        except Exception as e:
            logger.exception("Text encoding failed")
            raise AnalysisFailedError(
                message="Text encoding process failed"
            ) from e

class ImageEncoder(nn.Module):
    """Image encoder using ResNet backbone."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        logger.info(f"Loading model: {self.model_name}")

        try:
            if self.model_name == "resnet18":
                base_model = tv_models.resnet18(
                    weights=tv_models.ResNet18_Weights.DEFAULT
                )
                feature_dim = 512
            elif self.model_name == "resnet50":
                base_model = tv_models.resnet50(
                    weights=tv_models.ResNet50_Weights.DEFAULT
                )
                feature_dim = 2048
            else:
                raise InvalidInputError(
                    message=f"Unsupported image model: {self.model_name}"
                )
        except InvalidInputError:
            raise
        except Exception as e:
            logger.exception("Failed to load model")
            raise ModelNotLoadedError(
                message=f"Failed to load image encoder model '{self.model_name}'"
            ) from e
        
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_dim = feature_dim

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(DEVICE)
        self.encoder.eval()

        logger.info("Model loaded, parameters frozen")

    @torch.no_grad()
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode an image tensor to normalized feature vector

        Args:
            img_tensor: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Normalized feature vector of shape (B, feature_dim)
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise InvalidInputError(
                message="Image tensor must be a torch.Tensor"
            )
        
        image_tensor = image_tensor.to(DEVICE)

        try:
            features = self.encoder(image_tensor)
            features = torch.flatten(features, 1)
            features = torch.nn.functional.normalize(features, dim=-1)
            return features
        except Exception as e:
            logger.exception("Image encoding failed")
            raise AnalysisFailedError(
                message="Image encoding process failed"
            ) from e
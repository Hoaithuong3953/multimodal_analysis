"""Multimodal fusion model for late fusion architecture"""

from __future__ import annotations

import torch
import torch.nn as nn

class MultimodalFusionModel(nn.Module):
    """Late-fusion multimodal model with two prediction heads"""

    def __init__(self, text_dim: int = 768, image_dim: int = 512, hidden_dim: int = 256):
        """
        Initialize MultimodalFusionModel
        """
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim

        self.fusion = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.sentiment_head = nn.Linear(hidden_dim, 3)
        self.rebuy_head = nn.Linear(hidden_dim, 1)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: fuse embeddings and predict sentiment and repurchase

        Args:
            text_emb: Text embedding tensor of shape (B, text_dim)
            image_emb: Image embedding tensor of shape (B, image_dim)

        Returns:
            Tuple of (sentiment_logits, rebuy_logits):
                - sentiment_logits: (B, 3) logits for sentiment classes
                - rebuy_logits: (B, 1) logits for repurchase prediction
        """
        fused = torch.cat([text_emb, image_emb], dim=-1)
        h = self.fusion(fused)
        sent_logits = self.sentiment_head(h)
        rebuy_logits = self.rebuy_head(h)

        return sent_logits, rebuy_logits

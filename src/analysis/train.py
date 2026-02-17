"""Train the fusion model on synthetic data and save checkpoint"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from config import logger, settings
from src.analysis.encoders import TextEncoder, ImageEncoder, DEVICE
from src.analysis.pipeline import MultimodalFusionModel
from src.core import InvalidInputError

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

def _data_dir() -> Path:
    """Return the project data directory (multimodel_ai/data)"""
    return Path(__file__).resolve().parent.parent.parent / "data"


def _load_csv_rows(csv_path: Path, images_dir: Path) -> list[dict]:
    """
    Read a CSV file with columns text, image_path, sentiment, will_rebuy into a list of dicts.

    Args:
        csv_path: Path to the CSV file.
        images_dir: Base directory for resolving relative image paths.

    Returns:
        List of dicts with keys: text, image_path, sentiment, will_rebuy.
    """
    import csv
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = (r.get("text") or "").strip()
            if not text:
                continue
            image_path = (r.get("image_path") or "").strip()
            if image_path and images_dir.exists():
                full = images_dir / image_path
                image_path = str(full) if full.exists() else None
            else:
                image_path = None
            sentiment = int(r.get("sentiment", 1))
            will_rebuy = int(r.get("will_rebuy", 0))
            rows.append({"text": text, "image_path": image_path, "sentiment": sentiment, "will_rebuy": will_rebuy})
    return rows


def _load_synthetic_data() -> list[dict]:
    """
    Load training samples only from data/synthetic_list.csv

    Uses built-in sample list if the file is missing

    Returns:
        List of training samples (dicts with text, image_path, sentiment, will_rebuy)
    """
    data_dir = _data_dir()
    images_dir = data_dir / "images"
    csv_path = data_dir / "synthetic_list.csv"

    if not csv_path.exists():
        logger.warning("data/synthetic_list.csv not found, using built-in list")
        return _builtin_synthetic_data()

    rows = _load_csv_rows(csv_path, images_dir)
    logger.info(f"Loaded {len(rows)} samples from {csv_path}")
    return rows


def _split_train_val_test(
    data: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split data into train, validation, and test sets

    Args:
        data: List of sample dicts.
        train_ratio: Fraction of data for training (default 0.8)
        val_ratio: Fraction of data for validation (default 0.1)
        test_ratio: Fraction of data for test (default 0.1)
        seed: Random seed for reproducible shuffle

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], [], []
    n_train = max(0, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    train_data = shuffled[:n_train]
    val_data = shuffled[n_train : n_train + n_val]
    test_data = shuffled[n_train + n_val :]
    return train_data, val_data, test_data


def _eval_loss_accuracy(
    model: torch.nn.Module,
    data: list[dict],
    text_encoder: TextEncoder,
    image_encoder: ImageEncoder,
    ce: torch.nn.Module,
    bce: torch.nn.Module,
) -> tuple[float, float, float]:
    """
    Compute mean loss and accuracy (sentiment, will_rebuy) over a sample list

    Runs in eval mode with no gradient update

    Args:
        model: Fusion model
        data: List of sample dicts
        text_encoder: Text encoder
        image_encoder: Image encoder
        ce: Cross-entropy loss for sentiment
        bce: BCE loss for will_rebuy

    Returns:
        Tuple of (mean_loss, sentiment_accuracy, will_rebuy_accuracy)
    """
    if not data:
        return 0.0, 0.0, 0.0
    model.eval()
    total_loss = 0.0
    sent_correct = 0
    rebuy_correct = 0
    with torch.no_grad():
        for sample in data:
            text_emb = text_encoder(sample["text"]).unsqueeze(0)
            img_t = _load_image(sample.get("image_path"))
            image_emb = image_encoder(img_t)
            sent_logits, rebuy_logits = model(text_emb, image_emb)
            sent_target = torch.tensor([sample["sentiment"]], device=DEVICE, dtype=torch.long)
            rebuy_target = torch.tensor([[float(sample["will_rebuy"])]], device=DEVICE)
            loss = ce(sent_logits, sent_target) + bce(rebuy_logits, rebuy_target)
            total_loss += loss.item()
            sent_correct += (sent_logits.argmax(dim=1) == sent_target).sum().item()
            rebuy_correct += ((rebuy_logits > 0) == (rebuy_target > 0.5)).sum().item()
    n = len(data)
    return total_loss / n, sent_correct / n, rebuy_correct / n


def _builtin_synthetic_data() -> list[dict]:
    """
    Return built-in sample list when data/synthetic_list.csv is missing

    Returns:
        List of sample dicts for fallback training
    """
    images_dir = _data_dir() / "images"
    image_paths = list(images_dir.glob("*.png"))[:1] + list(images_dir.glob("*.jpg"))[:1]
    img_path: str | None = str(image_paths[0]) if image_paths else None
    return [
        {"text": "Hàng lỗi, đổi trả không được, rất thất vọng.", "image_path": img_path, "sentiment": 0, "will_rebuy": 0},
        {"text": "Sản phẩm tệ, chất lượng kém.", "image_path": None, "sentiment": 0, "will_rebuy": 0},
        {"text": "Sản phẩm rất tốt, giao hàng nhanh, sẽ ủng hộ lâu dài!", "image_path": img_path, "sentiment": 2, "will_rebuy": 1},
        {"text": "Tạm ổn, nhưng mà quá nhỏ.", "image_path": None, "sentiment": 1, "will_rebuy": 0},
        {"text": "Cho mình hỏi size áo này form rộng hay ôm?", "image_path": None, "sentiment": 1, "will_rebuy": 0},
    ]

if __name__ == "__main__":
    logger.info("Loading encoders (frozen)...")
    text_encoder = TextEncoder(model_name=settings.text_model_name)
    image_encoder = ImageEncoder(model_name=settings.image_model_name)
    model = MultimodalFusionModel(
        text_dim=text_encoder.hidden_size,
        image_dim=image_encoder.feature_dim,
    )
    model.to(DEVICE)

    data = _load_synthetic_data()
    
    if len(data) < 30:
        logger.warning(
            f"Dataset is very small ({len(data)} samples) - results may not be reliable. Consider collecting more data"
        )
    
    n_total = len(data)
    if n_total < 30:
        train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    else:
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    
    train_data, val_data, test_data = _split_train_val_test(
        data, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=42
    )
    logger.info(f"Split: train={len(train_data)} val={len(val_data)} test={len(test_data)}")
    
    if len(val_data) < 3:
        logger.warning(f"Validation set is very small ({len(val_data)} samples). Metrics may be unreliable")
    if len(test_data) < 3:
        logger.warning(f"Test set is very small ({len(test_data)} samples). Metrics may be unreliable")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()

    epochs = 50
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    out_path = Path(settings.checkpoint_path or "config/checkpoint.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for sample in train_data:
            text = sample["text"]
            image_path = sample.get("image_path")
            sentiment_idx = sample["sentiment"]
            will_rebuy = sample["will_rebuy"]

            with torch.no_grad():
                text_emb = text_encoder(text).unsqueeze(0)
                img_t = _load_image(image_path)
                image_emb = image_encoder(img_t)

            sent_logits, rebuy_logits = model(text_emb, image_emb)
            sent_target = torch.tensor([sentiment_idx], device=DEVICE, dtype=torch.long)
            rebuy_target = torch.tensor([[float(will_rebuy)]], device=DEVICE)

            loss_sent = ce(sent_logits, sent_target)
            loss_rebuy = bce(rebuy_logits, rebuy_target)
            loss = loss_sent + loss_rebuy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_data) if train_data else 0.0
        log_msg = f"Epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f}"

        if val_data:
            val_loss, val_sent_acc, val_rebuy_acc = _eval_loss_accuracy(
                model, val_data, text_encoder, image_encoder, ce, bce
            )
            scheduler.step(val_loss)
            log_msg += f" val_loss={val_loss:.4f} val_sent_acc={val_sent_acc:.2%} val_rebuy_acc={val_rebuy_acc:.2%}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), out_path)
                logger.info(f"  -> best val_loss, checkpoint saved to {out_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                    break

        logger.info(log_msg)

    if not val_data and train_data:
        torch.save(model.state_dict(), out_path)
        logger.info(f"Checkpoint saved to {out_path}")

    if test_data:
        test_loss, test_sent_acc, test_rebuy_acc = _eval_loss_accuracy(
            model, test_data, text_encoder, image_encoder, ce, bce
        )
        logger.info(
            f"Test: loss={test_loss:.4f} sentiment_acc={test_sent_acc:.2%} will_rebuy_acc={test_rebuy_acc:.2%}"
        )
# Multimodal AI – Sentiment & Will-rebuy

Multimodal analysis service: text + image → **sentiment** (negative / neutral / positive) and **will_rebuy** (0/1). Uses a fusion model on top of DistilBERT (text) and ResNet (image) encoders.

## Requirements

- Python 3.10+
- See `requirements.txt`: torch, torchvision, transformers, gradio, pillow, pydantic, pydantic-settings

## Setup

1. **Clone / go to project root** (folder containing `config/`, `src/`, `data/`):

   ```bash
   cd path/to/multimodel_ai
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Optional:** Copy `.env.example` to `.env` and set `CHECKPOINT_PATH`, `TEXT_MODEL_NAME`, `IMAGE_MODEL_NAME` if you need custom paths or model names. Defaults: `config/checkpoint.pt`, DistilBERT multilingual, ResNet18.

## Dataset

- **CSV:** `data/synthetic_list.csv`  
  Columns: `text`, `image_path`, `sentiment` (0=negative, 1=neutral, 2=positive), `will_rebuy` (0/1).
- **Images:** `data/images/`  
  Put the images referred to by `image_path` (e.g. `images/xxx.jpg`). Missing image is allowed (model uses zero image features).

Training loads **only** this CSV (no other list files).

## Training

From project root `multimodel_ai`:

```bash
python -m src.analysis.train
```

- Reads `data/synthetic_list.csv`, splits into train/val/test (80/10/10 or 60/20/20 if &lt; 30 samples).
- Saves best checkpoint by validation loss to `config/checkpoint.pt` (or `CHECKPOINT_PATH`).
- Logs train/val loss and accuracy; at the end prints **test loss**, **sentiment accuracy**, **will_rebuy accuracy**.

## Validation (F1 and accuracy)

After training, run validation on the **test set** (same split as training):

```bash
python -m scripts.validate
```

This script:

- Loads the checkpoint from `config/checkpoint.pt` (or `CHECKPOINT_PATH`).
- Evaluates on the test split and prints:
  - **Sentiment:** F1 (macro), accuracy. Requirement: F1 ≥ 0.75.
  - **Will_rebuy:** F1, accuracy. Requirement: F1 ≥ 0.65.
  - PASS/FAIL for each requirement.

If the checkpoint is missing, run training first: `python -m src.analysis.train`.

## Run the Gradio UI

From project root:

```bash
python -m src.main
```

This starts the Gradio demo (text + optional image input → sentiment and will_rebuy). The service loads the checkpoint from `config/checkpoint_path` if the file exists.

## Project structure (short)

| Path | Description |
|------|-------------|
| `config/settings.py` | App settings (checkpoint path, model names, logging). |
| `config/checkpoint.pt` | Saved fusion model weights (created by training). |
| `data/synthetic_list.csv` | Training/eval data (text, image_path, sentiment, will_rebuy). |
| `data/images/` | Image files used by the CSV. |
| `src/analysis/train.py` | Training script (entry: `python -m src.analysis.train`). |
| `src/analysis/pipeline.py` | Fusion model. |
| `src/analysis/encoders.py` | Text (DistilBERT) and image (ResNet) encoders. |
| `src/analysis/service.py` | Analysis service used by the API/UI. |
| `scripts/validate.py` | Validation script (F1/accuracy on test set). |
| `src/main.py` | Entry point for the Gradio UI. |
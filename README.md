## MediGraph

MediGraph is a handwritten medical text recognition project built around `microsoft/trocr-base-handwritten`.
The current focus is fine-tuning on doctor handwriting data and running local inference on prescription word images.

## Project Goal

- Fine-tune a TrOCR-based model on doctor handwritten medicine names.
- Track quality with loss and text metrics like CER/WER.
- Save best and final checkpoints for inference.

## Current Repository Layout

```text
MediGraph/
├── data/
│   ├── Training/
│   ├── Validation/
│   └── Testing/
├── logger/
│   └── logger.py
├── src/
│   ├── config.py
│   ├── inference.py
│   ├── model/
│   │   ├── dataloader.py
│   │   ├── definition.py
│   │   ├── preprocess.py
│   │   └── training.py
│   ├── pipelines/
│   │   └── training_pipeline.py
│   └── visualization/
│       └── plots.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Dataset Format

Expected split folders:

- `data/Training/training_words` + label file
- `data/Validation/validation_words` + label file
- `data/Testing/testing_words` + label file

Label files are expected to contain image file name and text label columns (CSV/XLSX supported in preprocessing flow).

## Environment Setup

### 1) Create and activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

Or with `uv`:

```powershell
uv sync
```

## Training

Current training entrypoint:

```powershell
python src/pipelines/training_pipeline.py
```

## Inference

`src/inference.py` exposes:

- `load_model(save_dir)`
- `predict(image_path, model, processor, device)`

Minimal usage example:

```python
from src.inference import load_model, predict

model, processor, device = load_model("path_to_saved_model")
predict("path_to_image.png", model, processor, device)
```

## Dependencies Used in Project

- `transformers`, `torch`, `torchvision`, `torchaudio`
- `peft`, `bitsandbytes`, `accelerate`
- `datasets`, `evaluate`, `jiwer`
- `pandas`, `kagglehub`, `matplotlib`
- `pillow`, `tqdm`, `fastapi`

## Notes

- Logging is configured in `logger/logger.py`.
- Visualization utilities are in `src/visualization/plots.py`.
- The codebase is in active development and training/inference modules are being iterated.

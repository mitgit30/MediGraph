## MediGraph

MediGraph is a prescription-understanding project with two core stages:
- OCR stage: fine-tuned TrOCR for doctor handwriting extraction.
- GraphRAG stage: DRKG + Neo4j retrieval for evidence-grounded medical reporting.

## Project Goal

- Extract text from handwritten prescriptions.
- Verify/normalize extracted text before downstream reasoning.
- Build a production-grade GraphRAG pipeline over DRKG.
- Generate a final structured LLM report with graph-backed evidence.

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
│   ├── graph/
│   │   ├── schema.cypher
│   │   └── ingest/
│   │       └── tsv_to_csv.py
│   ├── pipelines/
│   │   └── training_pipeline.py
│   └── visualization/
│       └── plots.py
├── drkg/
│   ├── drkg.tsv
│   ├── entity2src.tsv
│   └── relation_glossary.tsv
├── requirements.txt
├── pyproject.toml
└── README.md
```

## OCR Data Format

Expected split folders:

- `data/Training/training_words` + label file
- `data/Validation/validation_words` + label file
- `data/Testing/testing_words` + label file

Label files are expected to contain image file name and text label columns (CSV/XLSX supported in preprocessing flow).

## DRKG Assets Used

Required files for graph build:
- `drkg/drkg.tsv` (main edge list)
- `drkg/entity2src.tsv` (entity provenance)
- `drkg/relation_glossary.tsv` (relation metadata)

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

## OCR Training

Current training entrypoint:

```powershell
python src/pipelines/training_pipeline.py
```

## OCR Inference

`src/inference.py` exposes:

- `load_model(save_dir)`
- `predict(image_path, model, processor, device)`

Minimal usage example:

```python
from src.inference import load_model, predict

model, processor, device = load_model("path_to_saved_model")
predict("path_to_image.png", model, processor, device)
```

## DRKG TSV -> CSV Pipeline

Converter script:
- `src/graph/ingest/tsv_to_csv.py`

Run:

```powershell
python src/graph/ingest/tsv_to_csv.py
```

Default outputs:
- `artifacts/graph/import_csv/nodes.csv`
- `artifacts/graph/import_csv/edges.csv`
- `artifacts/graph/import_csv/relations.csv`
- `artifacts/graph/import_csv/entity_sources.csv`

Current conversion result:
- Edge rows read: `5,874,261`
- Edge rows written: `5,874,261`
- Bad edge rows: `0`
- Node rows written: `97,238`
- Relation rows written: `107`
- Entity source rows: `97,238`

## Neo4j Plan (Docker-first)

- Neo4j will be run in Docker for dev/prod consistency.
- Graph data should be mounted to a persistent volume (`/data`), so `docker compose stop/down` does not lose DB contents.
- Data loss happens only if volume is removed (for example `docker compose down -v`).

## GraphRAG Implementation Direction

Planned architecture:
1. Prescription upload (Streamlit).
2. TrOCR extraction.
3. Text verification/normalization.
4. Entity linking to graph.
5. Neo4j retrieval (GraphRAG).
6. Final LLM report with evidence.

Current stage:
- DRKG preprocessing completed.
- Neo4j schema/load/retrieval integration is the next implementation step.

## Dependencies Used in Project

- `transformers`, `torch`, `torchvision`, `torchaudio`
- `peft`, `bitsandbytes`, `accelerate`
- `datasets`, `evaluate`, `jiwer`
- `pandas`, `kagglehub`, `matplotlib`
- `pillow`, `tqdm`, `fastapi`

## Notes

- Logging is configured in `logger/logger.py`.
- Visualization utilities are in `src/visualization/plots.py`.
- The codebase is in active development and the GraphRAG stack is being built incrementally.

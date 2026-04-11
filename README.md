## MediGraph

MediGraph is a prescription understanding system with:
- OCR pipeline for doctor handwriting extraction (TrOCR fine-tuning).
- GraphRAG pipeline on DRKG + Neo4j for evidence-grounded retrieval.

## Current Project State

- OCR training/inference pipeline is available.
- DRKG preprocessing is completed (`tsv -> csv`).
- Neo4j ingestion is completed (graph loaded and queryable).
- Phase 2 retrieval modules are implemented:
  - entity linking
  - local graph retrieval
  - context building
  - basic pipeline orchestration
- Medicine normalization bridge is added (brand -> generic expansion from dataset labels).

## Data Sources Used

- OCR dataset: doctor handwritten prescription words (`data/Training`, `data/Validation`, `data/Testing`).
- Graph dataset:
  - `drkg/drkg.tsv`
  - `drkg/entity2src.tsv`
  - `drkg/relation_glossary.tsv`

## Quick Run Guide

### 1) Install dependencies

```powershell
pip install -r requirements.txt
```

### 2) Start Neo4j

```powershell
docker compose up -d
```

### 3) DRKG conversion (if needed)

```powershell
python src/graph/ingest/tsv_to_csv.py
```

### 4) Load graph into Neo4j

```powershell
python src/graph/ingest/load_neo4j.py
```

### 5) Run tests

```powershell
pytest -s tests/test_neo4j_client.py
pytest -s tests/test_medicine_mapper.py
pytest -s tests/test_graph_retrieval.py
```

## Important Outputs

Generated graph CSV artifacts:
- `artifacts/graph/import_csv/nodes.csv`
- `artifacts/graph/import_csv/edges.csv`
- `artifacts/graph/import_csv/relations.csv`
- `artifacts/graph/import_csv/entity_sources.csv`

## Environment

Use `.env` for runtime secrets/config.
Use `.env.example` as template.

## Next Steps

- Improve entity linking quality with stronger alias/synonym mapping.
- Add retrieval ranking and confidence calibration.
- Connect retrieval context to final LLM report generation.
- Integrate full flow in Streamlit UI.

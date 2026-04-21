## MediGraph

MediGraph is a prescription understanding system with:
- OCR pipeline for doctor handwriting extraction (TrOCR fine-tuning).
- GraphRAG pipeline on DRKG + Neo4j for evidence-grounded retrieval.

## Current Project State

The prototype now provides a **fully functional end‑to‑end pipeline**:

* **OCR stage** – training and inference code are operational and can process handwritten prescription images.
* **DRKG preprocessing** – TSV files are converted to CSV (`tsv → csv`) and are ready for bulk import.
* **Neo4j ingestion** – the graph schema is applied, CSVs are loaded, and the database is queryable via the provided client wrapper.
* **Retrieval layer (Phase 2)** – all core modules are in place:
  * **Entity linking** with exact, prefix, and fuzzy matching, plus the existing `MedicineMapper` for brand‑to‑generic expansion.
  * **Graph retriever** that fetches 1‑2‑hop neighborhoods around matched entities.
  * **Context builder** that formats retrieved evidence for downstream LLM consumption.
  * **Pipeline orchestration** that ties the above steps together and returns a structured result.
* **Alias/medicine mapper** – expands drug brand names to their generic identifiers, enabling more robust matching.
* **Test coverage** – the test suite validates the Neo4j client, medicine mapper, and retrieval pipeline; all tests pass when a Neo4j instance is available.
* **Configuration‑driven** – runtime settings (Neo4j credentials, LLM provider, batch sizes, etc.) are supplied via a `.env` file; no secrets are hard‑coded.

## Technical Architecture (Current)

1. OCR stage:
- `src/pipelines/training_pipeline.py` for model fine-tuning.
- `src/inference.py` for loading and predicting OCR text.

2. Graph build stage:
- `src/graph/ingest/tsv_to_csv.py` converts DRKG TSV files to import CSVs.
- `src/graph/ingest/load_neo4j.py` applies schema and ingests CSVs in batches.
- `src/graph/schema.cypher` defines Neo4j constraints/indexes.

3. Retrieval stage:
- `src/graph/retrieval/entity_linker.py` maps query terms to graph entities.
- `src/graph/retrieval/medicine_mapper.py` expands brand -> generic from label CSVs.
- `src/graph/retrieval/graph_retriever.py` retrieves local graph paths (1-2 hops).
- `src/graph/retrieval/context_builder.py` creates LLM-ready evidence context.
- `src/graph/retrieval/pipeline.py` orchestrates end-to-end retrieval flow.

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

## Runtime Configuration

Key variables in `.env`:
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- `GRAPH_INGEST_BATCH_SIZE`
- `GRAPH_ARTIFACTS_DIR`
- `OLLAMA_LLM_PROVIDER`, `OLLAMA_LLM_MODEL`, `OLLAMA_LLM_BASE_URL`, `OLLAMA_API_KEY`
- `GRAPH_MEDICINE_MAP_TRAIN`, `GRAPH_MEDICINE_MAP_VAL`, `GRAPH_MEDICINE_MAP_TEST`

## Important Outputs

Generated graph CSV artifacts:
- `artifacts/graph/import_csv/nodes.csv`
- `artifacts/graph/import_csv/edges.csv`
- `artifacts/graph/import_csv/relations.csv`
- `artifacts/graph/import_csv/entity_sources.csv`

## Environment

Use `.env` for runtime secrets/config.
Use `.env.example` as template.

## Neo4j Verification (Post-Ingestion)

Run in Neo4j Browser:

```cypher
MATCH (n:Entity) RETURN count(n) AS total_nodes;
MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS total_edges;
MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
RETURN a.id, r.relation_name, b.id
LIMIT 25;
```

Expected behavior:
- Node/edge counts are non-zero and aligned with ingestion logs.
- `relation_name` appears on relationships.
- `sources` appears on nodes where entity provenance exists.

## Retrieval Notes

- Retrieval is local-search oriented for current phase (anchor entities -> neighborhood paths).
- Matching uses `exact -> startswith -> contains` scoring.
- For real prescriptions, generic-name normalization is required before graph lookup.

## Next Steps

- Improve entity linking quality with stronger alias/synonym mapping.
- Add retrieval ranking and confidence calibration.
- Connect retrieval context to final LLM report generation.
- Integrate full flow in Streamlit UI.

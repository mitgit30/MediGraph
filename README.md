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
### Overall Project State

The repository now provides a **complete end-to-end prototype** that can:

1. **Ingest** the DRKG biomedical knowledge graph into Neo4j (schema applied, CSV import automated).
2. **Link** free-text prescription statements to graph entities using exact, prefix and fuzzy matching, with a medicine-specific alias mapper.
3. **Retrieve** a localized sub-graph (1-2 hops) around the matched entities, assemble evidence paths, and surface them as LLM-ready context.
4. **Generate** a concise natural-language summary via the configured LLM adapter (e.g., Ollama, OpenAI).
5. **Expose** a simple CLI entry point (`main.py`) for quick sanity checks and a clear path to wrap the flow in a FastAPI/Streamlit UI.

All components are wired together through thin adapters (`Neo4jClient`, `LocalLLMAdapter`) and respect the layered architecture described in the Technical Architecture section. Tests cover the Neo4j client, medicine mapper, and graph retrieval pipeline, confirming that each stage returns non-empty results when proper data is present.

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


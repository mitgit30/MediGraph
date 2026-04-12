import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

@dataclass
class GraphConfig:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    ingest_batch_size: int = int(os.getenv("GRAPH_INGEST_BATCH_SIZE", "5000"))
    query_timeout_seconds: int = int(os.getenv("GRAPH_QUERY_TIMEOUT_SECONDS", "30"))
    artifacts_dir: Path = Path(
        os.getenv("GRAPH_ARTIFACTS_DIR", "artifacts/graph/import_csv")
    )
    local_llm_provider: str = os.getenv("OLLAMA_LLM_PROVIDER", "ollama")
    local_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:120b-cloud")
    local_llm_base_url: str = os.getenv(
        "OLLAMA_LLM_BASE_URL", "https://ollama.com/"
    )
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "")
    medicine_map_files: tuple[Path, ...] = (
        Path(os.getenv("GRAPH_MEDICINE_MAP_TRAIN", "data/Training/training_labels.csv")),
        Path(os.getenv("GRAPH_MEDICINE_MAP_VAL", "data/Validation/validation_labels.csv")),
        Path(os.getenv("GRAPH_MEDICINE_MAP_TEST", "data/Testing/testing_labels.csv")),
    )

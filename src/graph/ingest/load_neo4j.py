import csv
from pathlib import Path
from typing import Dict, Iterator, List

from logger.logger import get_logger
from src.graph.config import GraphConfig
from src.graph.runtime.neo4j_client import Neo4jClient

logger = get_logger()


def _chunk_csv_rows(file_path: Path, batch_size: int) -> Iterator[List[Dict[str, str]]]:
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        batch: List[Dict[str, str]] = []
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def _safe_label(raw_type: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in (raw_type or "").strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        return "UnknownType"
    if cleaned[0].isdigit():
        cleaned = f"T_{cleaned}"
    return cleaned


def apply_schema(client: Neo4jClient, schema_path: Path) -> None:
    logger.info("Applying schema from %s", schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    schema_text = schema_path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in schema_text.split(";") if stmt.strip()]
    with client.driver.session(database=client.config.neo4j_database) as session:
        for statement in statements:
            session.run(statement)
    logger.info("Schema applied successfully.")


def load_nodes(client: Neo4jClient, nodes_path: Path, batch_size: int) -> int:
    logger.info("Loading nodes from %s with batch size %d", nodes_path, batch_size)
    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes.csv not found: {nodes_path}")

    query = """
    UNWIND $rows AS row
    MERGE (n:Entity {id: row.id})
    SET n.name = row.name,
        n.type = row.type
    WITH n, row
    CALL apoc.create.addLabels(n, [row.type_label]) YIELD node
    RETURN count(node) AS updated
    """

    total = 0
    with client.driver.session(database=client.config.neo4j_database) as session:
        for idx, batch in enumerate(_chunk_csv_rows(nodes_path, batch_size), start=1):
            payload = []
            for row in batch:
                payload.append(
                    {
                        "id": row.get("id", "").strip(),
                        "name": row.get("name", "").strip(),
                        "type": row.get("type", "").strip(),
                        "type_label": _safe_label(row.get("type", "")),
                    }
                )

            session.run(query, rows=payload).consume()
            total += len(payload)
            if idx % 20 == 0:
                logger.info("Nodes batch %d loaded. Total nodes processed: %d", idx, total)

    logger.info("Node load complete. Total nodes processed: %d", total)
    return total


def load_edges(client: Neo4jClient, edges_path: Path, batch_size: int) -> int:
    logger.info("Loading edges from %s with batch size %d", edges_path, batch_size)
    if not edges_path.exists():
        raise FileNotFoundError(f"edges.csv not found: {edges_path}")

    query = """
    UNWIND $rows AS row
    MATCH (h:Entity {id: row.head_id})
    MATCH (t:Entity {id: row.tail_id})
    MERGE (h)-[r:RELATED_TO {relation_name: row.relation_name}]->(t)
    SET r.head_type = row.head_type,
        r.tail_type = row.tail_type
    RETURN count(r) AS updated
    """

    total = 0
    with client.driver.session(database=client.config.neo4j_database) as session:
        for idx, batch in enumerate(_chunk_csv_rows(edges_path, batch_size), start=1):
            payload = []
            for row in batch:
                payload.append(
                    {
                        "head_id": row.get("head_id", "").strip(),
                        "tail_id": row.get("tail_id", "").strip(),
                        "relation_name": row.get("relation", "").strip(),
                        "head_type": row.get("head_type", "").strip(),
                        "tail_type": row.get("tail_type", "").strip(),
                    }
                )

            session.run(query, rows=payload).consume()
            total += len(payload)
            if idx % 20 == 0:
                logger.info("Edges batch %d loaded. Total edges processed: %d", idx, total)

    logger.info("Edge load complete. Total edges processed: %d", total)
    return total


def load_entity_sources(client: Neo4jClient, entity_sources_path: Path, batch_size: int) -> int:
    logger.info("Loading entity sources from %s", entity_sources_path)
    if not entity_sources_path.exists():
        logger.warning("entity_sources.csv not found, skipping source enrichment.")
        return 0

    query = """
    UNWIND $rows AS row
    MATCH (n:Entity {id: row.entity_id})
    SET n.sources = row.sources
    RETURN count(n) AS updated
    """

    total = 0
    with client.driver.session(database=client.config.neo4j_database) as session:
        for batch in _chunk_csv_rows(entity_sources_path, batch_size):
            payload = [
                {
                    "entity_id": row.get("entity_id", "").strip(),
                    "sources": row.get("sources", "").strip(),
                }
                for row in batch
            ]
            session.run(query, rows=payload).consume()
            total += len(payload)

    logger.info("Entity source enrichment complete. Total records processed: %d", total)
    return total


def main() -> None:
    config = GraphConfig()
    client = Neo4jClient(config)

    import_dir = config.artifacts_dir
    schema_path = Path("src/graph/schema.cypher")
    nodes_path = import_dir / "nodes.csv"
    edges_path = import_dir / "edges.csv"
    entity_sources_path = import_dir / "entity_sources.csv"

    try:
        apply_schema(client, schema_path)
    except Exception as exc:
        logger.exception("Schema apply failed.")
        client.close()
        raise RuntimeError("Failed while applying Neo4j schema.") from exc

    try:
        load_nodes(client, nodes_path, config.ingest_batch_size)
        
    except Exception as exc:
        logger.exception("Node load failed.")
        client.close()
        raise RuntimeError("Failed while loading nodes.") from exc

    try:
        load_edges(client, edges_path, config.ingest_batch_size)
    except Exception as exc:
        logger.exception("Edge load failed.")
        client.close()
        raise RuntimeError("Failed while loading edges.") from exc

    try:
        load_entity_sources(client, entity_sources_path, config.ingest_batch_size)
    except Exception as exc:
        logger.exception("Entity source load failed.")
        client.close()
        raise RuntimeError("Failed while loading entity sources.") from exc

    client.close()
    logger.info("Neo4j ingestion completed successfully.")


if __name__ == "__main__":
    main()

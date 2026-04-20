from pathlib import Path

from logger.logger import get_logger
from src.graph.config import GraphConfig
from src.graph.runtime.neo4j_client import Neo4jClient

logger = get_logger()


def _to_neo4j_file_url(file_path: Path) -> str:
    return f"file:///{file_path.name}"


def apply_schema(client: Neo4jClient, schema_path: Path) -> None:
    logger.info("Applying schema from %s", schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    schema_text = schema_path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in schema_text.split(";") if stmt.strip()]
    with client.driver.session(database=client.config.neo4j_database) as session:
        for statement in statements:
            session.run(statement)
        session.run("CALL db.awaitIndexes()")
    logger.info("Schema applied successfully.")


def load_nodes(client: Neo4jClient, nodes_path: Path, batch_size: int) -> int:
    logger.info("Loading nodes from %s with batch size %d", nodes_path, batch_size)
    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes.csv not found: {nodes_path}")

    query = """
    CALL apoc.periodic.iterate(
      "LOAD CSV WITH HEADERS FROM $csv_url AS row RETURN row",
      "
        MERGE (n:Entity {id: row.id})
        SET
            n.name = coalesce(nullIf(row.mapped_name, ''), row.name),
            n.raw_name = row.name,
            n.type = row.type,
            n.clean_id = row.clean_id,
            n.symbol = row.symbol
        WITH n, row
        WITH n,
             CASE
                 WHEN coalesce(row.type_label, '') <> '' THEN row.type_label
                 ELSE apoc.text.regreplace(coalesce(row.type, 'UnknownType'), '[^A-Za-z0-9]', '_')
             END AS raw_label
        WITH n,
             CASE
                 WHEN raw_label =~ '^[0-9].*' THEN 'T_' + raw_label
                 ELSE raw_label
             END AS type_label
        CALL apoc.create.addLabels(n, [type_label]) YIELD node
        RETURN count(*)
      ",
      {batchSize: $batch_size, parallel: true, params: {csv_url: $csv_url}}
    )
    YIELD total
    RETURN total
    """

    csv_url = _to_neo4j_file_url(nodes_path)
    with client.driver.session(database=client.config.neo4j_database) as session:
        record = session.run(query, csv_url=csv_url, batch_size=batch_size).single()
    total = int(record["total"]) if record and record.get("total") is not None else 0
    logger.info("Node load complete. Total nodes processed: %d", total)
    return total


def load_edges(client: Neo4jClient, edges_path: Path, batch_size: int) -> int:
    logger.info("Loading edges from %s with batch size %d", edges_path, batch_size)
    if not edges_path.exists():
        raise FileNotFoundError(f"edges.csv not found: {edges_path}")

    query = """
    CALL apoc.periodic.iterate(
      "LOAD CSV WITH HEADERS FROM $csv_url AS row RETURN row",
      "
        MATCH (h:Entity {id: row.head_id})
        MATCH (t:Entity {id: row.tail_id})
        MERGE (h)-[r:RELATED_TO {relation_name: coalesce(row.relation_name, row.relation)}]->(t)
        SET
            r.relation_type = row.relation_type,
            r.head_type = row.head_type,
            r.tail_type = row.tail_type,
            r.sources = row.sources
        RETURN count(*)
      ",
      {batchSize: $batch_size, parallel: true, params: {csv_url: $csv_url}}
    )
    YIELD total
    RETURN total
    """

    csv_url = _to_neo4j_file_url(edges_path)
    with client.driver.session(database=client.config.neo4j_database) as session:
        record = session.run(query, csv_url=csv_url, batch_size=batch_size).single()
    total = int(record["total"]) if record and record.get("total") is not None else 0
    logger.info("Edge load complete. Total edges processed: %d", total)
    return total


def load_entity_sources(client: Neo4jClient, entity_sources_path: Path, batch_size: int) -> int:
    logger.info("Loading entity sources from %s", entity_sources_path)
    if not entity_sources_path.exists():
        logger.warning("entity_sources.csv not found, skipping source enrichment.")
        return 0

    query = """
    CALL apoc.periodic.iterate(
      "LOAD CSV WITH HEADERS FROM $csv_url AS row RETURN row",
      "
        MATCH (n:Entity {id: row.entity_id})
        SET
            n.sources = row.sources,
            n.source_systems = [value IN split(coalesce(row.source_systems, ''), '|') WHERE trim(value) <> '']
        RETURN count(*)
      ",
      {batchSize: $batch_size, parallel: true, params: {csv_url: $csv_url}}
    )
    YIELD total
    RETURN total
    """

    csv_url = _to_neo4j_file_url(entity_sources_path)
    with client.driver.session(database=client.config.neo4j_database) as session:
        record = session.run(query, csv_url=csv_url, batch_size=batch_size).single()
    total = int(record["total"]) if record and record.get("total") is not None else 0
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

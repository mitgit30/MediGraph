import os
import unittest

from src.graph.config import GraphConfig
from src.graph.runtime.neo4j_client import Neo4jClient


def test_neo4j():
    config = GraphConfig()
    print(f"[test_neo4j] URI={config.neo4j_uri} DB={config.neo4j_database} USER={config.neo4j_user}")
    if not config.neo4j_password:
        print("[test_neo4j] Skipping: NEO4J_PASSWORD is not set.")
        return

    client = Neo4jClient(config)

    assert client.ping()
    print("[test_neo4j] Ping success.")

    with client.driver.session(database=config.neo4j_database) as session:
        result = session.run("RETURN 'ok' AS status").single()

    assert result["status"] == "ok"
    print(f"[test_neo4j] Query status={result['status']}")

    client.close()
    print("[test_neo4j] Client closed.")
    
test_neo4j()

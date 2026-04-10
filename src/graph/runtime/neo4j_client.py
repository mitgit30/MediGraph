from neo4j import GraphDatabase

from src.graph.config import GraphConfig


class Neo4jClient:
    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def ping(self) -> bool:
        query = "RETURN 1 AS ok"
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query)
            record = result.single()
            return bool(record and record["ok"] == 1)

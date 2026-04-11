from typing import Dict, List

from logger.logger import get_logger
from src.graph.runtime.neo4j_client import Neo4jClient

from .models import EvidencePath

logger = get_logger()


class GraphRetriever:
    def __init__(self, neo4j_client: Neo4jClient) -> None:
        self.neo4j_client = neo4j_client

    def get_local_paths(
        self,
        anchor_entity_ids: List[str],
        max_hops: int = 1,
        max_paths_per_entity: int = 25,
    ) -> List[EvidencePath]:
        if max_hops < 1:
            max_hops = 1
        if max_hops > 2:
            max_hops = 2

        hop_pattern = f"1..{max_hops}"
        query = f"""
        MATCH p=(a:Entity {{id: $anchor_id}})-[:RELATED_TO*{hop_pattern}]->(b:Entity)
        RETURN [n IN nodes(p) | {{id: n.id, name: n.name, type: n.type}}] AS nodes,
               [r IN relationships(p) | {{
                   relation_name: r.relation_name,
                   head_type: r.head_type,
                   tail_type: r.tail_type
               }}] AS relationships
        LIMIT $limit
        """

        all_paths: List[EvidencePath] = []
        try:
            with self.neo4j_client.driver.session(
                database=self.neo4j_client.config.neo4j_database
            ) as session:
                for anchor_id in anchor_entity_ids:
                    records = session.run(
                        query,
                        anchor_id=anchor_id,
                        limit=max_paths_per_entity,
                    )
                    for row in records:
                        all_paths.append(
                            EvidencePath(
                                anchor_entity_id=anchor_id,
                                nodes=list(row["nodes"]),
                                relationships=list(row["relationships"]),
                            )
                        )
            logger.info(
                "GraphRetriever returned %d paths for %d anchors.",len(all_paths),len(anchor_entity_ids))
            
            return all_paths
        
        except Exception as exc:
            
            logger.exception("Graph retrieval failed.")
            raise RuntimeError("Failed to retrieve graph paths.") from exc

    @staticmethod
    def flatten_unique_edges(paths: List[EvidencePath]) -> List[Dict[str, str]]:
        edges = []
        seen = set()

        for path in paths:
            nodes = path.nodes
            rels = path.relationships
            for idx, rel in enumerate(rels):
                if idx + 1 >= len(nodes):
                    continue
                head_id = nodes[idx].get("id", "")
                tail_id = nodes[idx + 1].get("id", "")
                key = (head_id, rel.get("relation_name", ""), tail_id)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    {
                        "head_id": head_id,
                        "relation_name": rel.get("relation_name", ""),
                        "tail_id": tail_id,
                        "head_type": rel.get("head_type", ""),
                        "tail_type": rel.get("tail_type", ""),
                    }
                )
        return edges

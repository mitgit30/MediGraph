from typing import Dict

from logger.logger import get_logger
from src.graph.config import GraphConfig
from src.graph.runtime.neo4j_client import Neo4jClient

from .context_builder import ContextBuilder 
from .entity_linker import EntityLinker
from .graph_retriever import GraphRetriever
from .medicine_mapper import MedicineMapper
from .models import RetrievalResult

logger = get_logger()


class GraphRetrievalPipeline:
    def __init__(self, neo4j_client: Neo4jClient) -> None:
        graph_config = GraphConfig()
        medicine_mapper = MedicineMapper(graph_config.medicine_map_files)
        self.entity_linker = EntityLinker(neo4j_client, medicine_mapper=medicine_mapper)
        self.graph_retriever = GraphRetriever(neo4j_client)
        self.context_builder = ContextBuilder()

    def run(
        self,
        query_text: str,
        max_terms: int = 20,
        top_k_per_term: int = 5,
        max_hops: int = 1,
        max_paths_per_entity: int = 25,
        max_anchor_entities: int = 20,
    ) -> Dict[str, object]:
        try:
            matches = self.entity_linker.link_text(
                query_text,
                max_terms=max_terms,
                top_k_per_term=top_k_per_term,
            )
            anchor_ids = []
            for match in matches:
                if match.entity_id in anchor_ids:
                    continue
                anchor_ids.append(match.entity_id)
                if len(anchor_ids) >= max_anchor_entities:
                    break
            paths = self.graph_retriever.get_local_paths(
                anchor_entity_ids=anchor_ids,
                max_hops=max_hops,
                max_paths_per_entity=max_paths_per_entity,
            )
            edges = self.graph_retriever.flatten_unique_edges(paths)

            result = RetrievalResult(
                query_text=query_text,
                entity_matches=matches,
                evidence_paths=paths,
                evidence_edges=edges,
            )
            context = self.context_builder.build(query_text, matches, paths)
            logger.info(
                "GraphRetrievalPipeline completed with %d matches, %d paths.",
                len(matches),
                len(paths),
            )
            return {"result": result, "context": context}
        except Exception as exc:
            logger.exception("Graph retrieval pipeline failed.")
            raise RuntimeError("Failed to run graph retrieval pipeline.") from exc

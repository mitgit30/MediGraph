from src.graph.config import GraphConfig
from src.graph.retrieval.pipeline import GraphRetrievalPipeline
from src.graph.runtime.neo4j_client import Neo4jClient


def test_graph():
    config = GraphConfig()
    if not config.neo4j_password:
        print("test_graph Skipping: NEO4J_PASSWORD is not set.")
        return

    client = Neo4jClient(config)
    pipeline = GraphRetrievalPipeline(client)

    query_text = "Gene::2157"
    output = pipeline.run(query_text=query_text,max_terms=10,top_k_per_term=3,max_hops=1,max_paths_per_entity=5)
    result = output["result"]
    context = output["context"]

    print(f"test_graph matches={len(result.entity_matches)}")
    print(f"test_graph paths={len(result.evidence_paths)}")
    print(f"test_graph edges={len(result.evidence_edges)}")
    print(f"test_graph evidence_lines={len(context['evidence_lines'])}")

    assert len(result.entity_matches) > 0 # check there are matched entities
    assert len(result.evidence_paths) > 0 # check graph paths are returned
    assert isinstance(context["summary"], str) # Check if summary is a string 
    assert isinstance(context["evidence_lines"], list) # Check if evidence lines is a list
    assert isinstance(context["citations"], list) # check if citations generated is a list

    client.close()

test_graph()

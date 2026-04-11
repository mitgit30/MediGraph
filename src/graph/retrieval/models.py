from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EntityMatch:
    query_term: str
    entity_id: str
    name: str
    entity_type: str
    score: float
    match_reason: str
    sources: str = ""


@dataclass
class EvidencePath:
    anchor_entity_id: str
    nodes: List[Dict[str, str]] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query_text: str
    entity_matches: List[EntityMatch] = field(default_factory=list)
    evidence_paths: List[EvidencePath] = field(default_factory=list)
    evidence_edges: List[Dict[str, str]] = field(default_factory=list)

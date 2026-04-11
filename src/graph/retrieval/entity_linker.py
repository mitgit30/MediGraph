import re
from typing import List

from logger.logger import get_logger
from src.graph.runtime.neo4j_client import Neo4jClient

from .medicine_mapper import MedicineMapper
from .models import EntityMatch

logger = get_logger()


class EntityLinker:
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        medicine_mapper: MedicineMapper | None = None,
    ) -> None:
        self.neo4j_client = neo4j_client
        self.medicine_mapper = medicine_mapper

    def _extract_terms(self, text: str, max_terms: int = 20) -> List[str]:
        chunks = re.split(r"[\n,;/]+", text or "")
        terms: List[str] = []
        stop_terms = {"tab", "tablet", "cap", "capsule", "mg", "ml", "od", "bd", "hs"}

        for chunk in chunks:
            term = re.sub(r"\s+", " ", chunk.strip())
            if len(term) < 3:
                continue
            terms.append(term)
            for token in term.split(" "):
                token = token.strip()
                if len(token) >= 3 and token.lower() not in stop_terms:
                    terms.append(token)
                alnum = re.sub(r"[^A-Za-z0-9:+\-]", "", token)
                if len(alnum) >= 3 and alnum.lower() not in stop_terms:
                    terms.append(alnum)

        if not terms:
            words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+]{2,}", text or "")
            terms = words

        seen = set()
        unique_terms = []
        for term in terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_terms.append(term)

        return unique_terms[:max_terms]

    def _find_matches_for_term(self, term: str, limit: int = 5) -> List[EntityMatch]:
        exact_query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) = toLower($term) OR toLower(n.id) = toLower($term)
        RETURN n.id AS entity_id, n.name AS name, n.type AS entity_type, n.sources AS sources
        LIMIT $limit
        """
        startswith_query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) STARTS WITH toLower($term)
           OR toLower(n.id) STARTS WITH toLower($term)
        RETURN n.id AS entity_id, n.name AS name, n.type AS entity_type, n.sources AS sources
        LIMIT $limit
        """
        contains_query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($term)
           OR toLower(n.id) CONTAINS toLower($term)
        RETURN n.id AS entity_id, n.name AS name, n.type AS entity_type, n.sources AS sources
        LIMIT $limit
        """

        results: List[EntityMatch] = []
        with self.neo4j_client.driver.session(
            database=self.neo4j_client.config.neo4j_database
        ) as session:
            exact_rows = list(session.run(exact_query, term=term, limit=limit))
            for row in exact_rows:
                results.append(
                    EntityMatch(
                        query_term=term,
                        entity_id=row["entity_id"],
                        name=row["name"] or "",
                        entity_type=row["entity_type"] or "",
                        score=1.0,
                        match_reason="exact",
                        sources=row["sources"] or "",
                    )
                )

            if results:
                return results

            startswith_rows = list(session.run(startswith_query, term=term, limit=limit))
            for row in startswith_rows:
                results.append(
                    EntityMatch(
                        query_term=term,
                        entity_id=row["entity_id"],
                        name=row["name"] or "",
                        entity_type=row["entity_type"] or "",
                        score=0.85,
                        match_reason="startswith",
                        sources=row["sources"] or "",
                    )
                )

            if results:
                return results

            contains_rows = list(session.run(contains_query, term=term, limit=limit))
            for row in contains_rows:
                results.append(
                    EntityMatch(
                        query_term=term,
                        entity_id=row["entity_id"],
                        name=row["name"] or "",
                        entity_type=row["entity_type"] or "",
                        score=0.6,
                        match_reason="contains",
                        sources=row["sources"] or "",
                    )
                )
        return results

    def link_text(self, text: str, max_terms: int = 20, top_k_per_term: int = 5) -> List[EntityMatch]:
        try:
            terms = self._extract_terms(text, max_terms=max_terms)
            if self.medicine_mapper:
                terms = self.medicine_mapper.expand_terms(terms, max_terms=max_terms * 3)
            logger.info("EntityLinker extracted %d terms.", len(terms))
            matches: List[EntityMatch] = []
            for term in terms:
                term_matches = self._find_matches_for_term(term, limit=top_k_per_term)
                matches.extend(term_matches)

            unique = {}
            for match in matches:
                key = match.entity_id
                if key not in unique or match.score > unique[key].score:
                    unique[key] = match

            final_matches = sorted(unique.values(), key=lambda x: x.score, reverse=True)
            logger.info("EntityLinker returned %d total matches.", len(final_matches))
            return final_matches
        except Exception as exc:
            logger.exception("Entity linking failed.")
            raise RuntimeError("Failed to link entities from text.") from exc

from typing import Dict, List

from logger.logger import get_logger

from .models import EntityMatch, EvidencePath

logger = get_logger()


class ContextBuilder:
    def build(
        self,
        query_text: str,
        matches: List[EntityMatch],
        paths: List[EvidencePath],
        max_evidence_lines: int = 40,
    ) -> Dict[str, object]:
        try:
            evidence_lines = self._build_evidence_lines(paths, max_evidence_lines)
            citations = self._build_citations(matches, max_evidence_lines)

            summary = (
                f"Query text length: {len(query_text)} | "
                f"Entity matches: {len(matches)} | Paths: {len(paths)}"
            )
            logger.info("ContextBuilder created %d evidence lines.", len(evidence_lines))
            return {
                "summary": summary,
                "evidence_lines": evidence_lines,
                "citations": citations,
            }
        except Exception as exc:
            logger.exception("Context build failed.")
            raise RuntimeError("Failed to build LLM context.") from exc

    def _build_evidence_lines(
        self,
        paths: List[EvidencePath],
        max_evidence_lines: int,
    ) -> List[str]:
        lines = []
        for path in paths:
            for idx, rel in enumerate(path.relationships):
                if idx + 1 >= len(path.nodes):
                    continue
                source = path.nodes[idx]
                target = path.nodes[idx + 1]
                lines.append(
                    f"{source.get('id', '')} --[{rel.get('relation_name', '')}]--> "
                    f"{target.get('id', '')}"
                )
                if len(lines) >= max_evidence_lines:
                    return lines
        return lines

    def _build_citations(
        self,
        matches: List[EntityMatch],
        max_citations: int,
    ) -> List[str]:
        citations = []
        for match in matches:
            source_text = match.sources if match.sources else "source-not-available"
            citations.append(f"{match.entity_id} | {source_text}")
            if len(citations) >= max_citations:
                break
        return citations

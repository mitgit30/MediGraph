import os
from typing import Dict, Iterable, List

from Bio import Entrez
from logger.logger import get_logger
from mygene import MyGeneInfo

logger = get_logger()


class GeneNameResolver:
    def __init__(self, chunk_size: int = 1000) -> None:
        self.chunk_size = chunk_size
        self.mg_client = MyGeneInfo()
        self.cache: Dict[str, str] = {}
        self.ncbi_email = os.getenv("NCBI_EMAIL", "").strip()
        if self.ncbi_email:
            Entrez.email = self.ncbi_email

    def resolve_symbols(self, gene_numbers: Iterable[str]) -> Dict[str, str]:
        numbers = sorted({item.strip() for item in gene_numbers if item and item.strip().isdigit()})
        if not numbers:
            return {}

        unresolved = [item for item in numbers if item not in self.cache]
        if unresolved:
            self._resolve_with_mygene(unresolved)

        remaining = [item for item in unresolved if item not in self.cache]
        if remaining:
            self._resolve_with_entrez(remaining)

        return {item: self.cache[item] for item in numbers if item in self.cache}

    def _resolve_with_mygene(self, gene_numbers: List[str]) -> None:
        for index in range(0, len(gene_numbers), self.chunk_size):
            batch = gene_numbers[index : index + self.chunk_size]
            try:
                results = self.mg_client.querymany(
                    batch,
                    scopes="entrezgene",
                    fields="symbol,name",
                    species="human",
                    as_dataframe=False,
                    returnall=False,
                )
            except Exception:
                logger.exception("mygene batch lookup failed for %d ids.", len(batch))
                continue

            for row in results:
                query = str(row.get("query", "")).strip()
                if not query or row.get("notfound"):
                    continue
                symbol = str(row.get("symbol", "")).strip()
                if symbol:
                    self.cache[query] = symbol

    def _resolve_with_entrez(self, gene_numbers: List[str]) -> None:
        if not self.ncbi_email:
            logger.warning("NCBI_EMAIL is not set. Skipping Biopython Entrez fallback.")
            return

        for gene_number in gene_numbers:
            try:
                with Entrez.esummary(db="gene", id=gene_number, retmode="xml") as handle:
                    summary = Entrez.read(handle)
                docs = summary.get("DocumentSummarySet", {}).get("DocumentSummary", [])
                if not docs:
                    continue
                symbol = str(docs[0].get("Name", "")).strip()
                if symbol:
                    self.cache[gene_number] = symbol
            except Exception:
                logger.exception("Entrez lookup failed for Gene::%s", gene_number)

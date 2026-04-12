import csv
from pathlib import Path
from typing import Dict, Iterable, List

from logger.logger import get_logger

logger = get_logger()


class MedicineMapper:
    def __init__(self, csv_paths: Iterable[Path]) -> None:
        self.csv_paths = list(csv_paths)
        self.brand_to_generic: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        loaded_rows = 0
        for csv_path in self.csv_paths:
            try:
                if not csv_path.exists():
                    logger.warning("Medicine map file not found: %s", csv_path)
                    continue

                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames:
                        continue
                    for row in reader:
                        brand = (row.get("MEDICINE_NAME") or "").strip()
                        generic = (row.get("GENERIC_NAME") or "").strip()
                        if not brand or not generic:
                            continue
                        self.brand_to_generic[brand.lower()] = generic
                        loaded_rows += 1
            except Exception as exc:
                logger.exception("Failed loading medicine map file: %s", csv_path)
                raise RuntimeError(f"Medicine map load failed for {csv_path}") from exc

        logger.info(
            "MedicineMapper loaded %d mappings (%d unique brands).",
            loaded_rows,
            len(self.brand_to_generic),
        )

    def expand_terms(self, terms: List[str], max_terms: int = 50) -> List[str]:
        expanded = list(terms)
        seen = {term.lower() for term in terms}

        for term in terms:
            generic = self.brand_to_generic.get(term.lower())
            if not generic:
                continue
            if generic.lower() not in seen:
                expanded.append(generic)
                seen.add(generic.lower())

        return expanded[:max_terms]

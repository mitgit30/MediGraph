import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re

from src.graph.ingest.gene_name_mapper import GeneNameResolver


@dataclass
class ParseStats:
    edge_rows_read: int = 0
    edge_rows_written: int = 0
    bad_edge_rows: int = 0
    node_rows_written: int = 0
    relation_rows_written: int = 0
    source_rows_written: int = 0


def parse_entity(raw_entity: str) -> tuple[str, str, str]:
    text = raw_entity.strip()
    if "::" in text:
        entity_type, entity_value = text.split("::", 1)
    else:
        entity_type, entity_value = "Unknown", text
    return text, entity_type.strip(), entity_value.strip()


def sanitize_relation_type(raw_relation: str) -> str:
    value = (raw_relation or "").strip()
    if not value:
        return "UNKNOWN_RELATION"
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = value.strip("_")
    return value.upper() if value else "UNKNOWN_RELATION"


def build_clean_id(entity_id: str) -> str:
    return "".join((entity_id or "").split())


def build_entity_name(
    entity_type: str,
    entity_name: str,
    entity_id: str,
    gene_symbol_map: dict[str, str] | None = None,
    compound_name_map: dict[str, str] | None = None,
) -> str:
    name = (entity_name or "").strip()
    if entity_type == "Compound":
        compound_key = name.upper()
        if compound_name_map and compound_key in compound_name_map:
            return compound_name_map[compound_key]
    if entity_type == "Gene" and name.isdigit():
        if gene_symbol_map and name in gene_symbol_map:
            return gene_symbol_map[name]
        return f"Gene {name}"
    if name:
        return name
    return entity_id


def build_base_name(entity_type: str, entity_name: str, entity_id: str) -> str:
    name = (entity_name or "").strip()
    if entity_type == "Gene" and name.isdigit():
        return f"Gene {name}"
    if name:
        return name
    return entity_id


def build_gene_symbol(entity_type: str, entity_name: str, gene_symbol_map: dict[str, str]) -> str:
    name = (entity_name or "").strip()
    if entity_type != "Gene":
        return ""
    if name.isdigit() and name in gene_symbol_map:
        return gene_symbol_map[name]
    if name and not name.isdigit():
        return name
    return ""


def build_source_systems(source_text: str) -> str:
    known_sources = (
        "DrugBank",
        "Hetionet",
        "STRING",
        "GNBR",
        "BioGRID",
        "IntAct",
        "DGIdb",
        "CTD",
        "KEGG",
        "PharmGKB",
        "Bioarx",
    )
    found: set[str] = set()
    source_lower = source_text.lower()
    for source in known_sources:
        if source.lower() in source_lower:
            found.add(source)
    return "|".join(sorted(found))


def read_entity_sources(entity2src_path: Path) -> dict[str, str]:
    sources_by_entity: dict[str, str] = {}
    with entity2src_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            entity_id = row[0].strip()
            if not entity_id:
                continue
            source_text = " | ".join(cell.strip() for cell in row[1:] if cell.strip())
            sources_by_entity[entity_id] = source_text
    return sources_by_entity


def read_relation_glossary(relation_glossary_path: Path) -> dict[str, dict[str, str]]:
    relation_info: dict[str, dict[str, str]] = {}
    with relation_glossary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            relation_name = (row.get("Relation-name") or "").strip()
            if not relation_name:
                continue
            relation_info[relation_name] = {
                "relation_name": relation_name,
                "data_source": (row.get("Data-source") or "").strip(),
                "connected_entity_types": (row.get("Connected entity-types") or "").strip(),
                "interaction_type": (row.get("Interaction-type") or "").strip(),
                "description": (row.get("Description") or "").strip(),
                "reference": (row.get("Reference for the description") or "").strip(),
            }
    return relation_info


def read_drugbank_vocabulary(drugbank_vocab_path: Path) -> dict[str, str]:
    compound_name_map: dict[str, str] = {}
    if not drugbank_vocab_path.exists():
        return compound_name_map

    with drugbank_vocab_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            drugbank_id = (row.get("DrugBank ID") or "").strip().upper()
            common_name = (row.get("Common name") or "").strip()
            if not drugbank_id or not common_name:
                continue
            compound_name_map[drugbank_id] = common_name
    return compound_name_map


def collect_gene_numbers(drkg_tsv_path: Path) -> set[str]:
    gene_numbers: set[str] = set()
    with drkg_tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != 3:
                continue
            head_id, head_type, head_name = parse_entity(row[0])
            tail_id, tail_type, tail_name = parse_entity(row[2])
            if head_type == "Gene" and head_name.isdigit() and head_id:
                gene_numbers.add(head_name)
            if tail_type == "Gene" and tail_name.isdigit() and tail_id:
                gene_numbers.add(tail_name)
    return gene_numbers


def safe_label(raw_type: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", (raw_type or "").strip()) if part]
    if not parts:
        return "UnknownType"
    cleaned = "".join(part[0].upper() + part[1:] for part in parts)
    if cleaned[0].isdigit():
        return f"T_{cleaned}"
    return cleaned


def convert_drkg_tsv_to_csv(
    drkg_tsv_path: Path,
    entity2src_path: Path,
    relation_glossary_path: Path,
    drugbank_vocab_path: Path,
    output_dir: Path,
    resolve_gene_names: bool = True,
) -> ParseStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_csv_path = output_dir / "edges.csv"
    nodes_csv_path = output_dir / "nodes.csv"
    relations_csv_path = output_dir / "relations.csv"
    entity_sources_csv_path = output_dir / "entity_sources.csv"

    stats = ParseStats()
    entity_sources = read_entity_sources(entity2src_path)
    relation_info = read_relation_glossary(relation_glossary_path)
    compound_name_map = read_drugbank_vocabulary(drugbank_vocab_path)
    gene_symbol_map: dict[str, str] = {}

    if resolve_gene_names:
        gene_numbers = collect_gene_numbers(drkg_tsv_path)
        resolver = GeneNameResolver()
        gene_symbol_map = resolver.resolve_symbols(gene_numbers)

    nodes_by_id: dict[str, tuple[str, str, str, str, str, str]] = {}

    with drkg_tsv_path.open("r", encoding="utf-8", newline="") as drkg_handle, edges_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as edges_handle:
        reader = csv.reader(drkg_handle, delimiter="\t")
        writer = csv.DictWriter(
            edges_handle,
            fieldnames=[
                "head_id",
                "head_type",
                "head_name",
                "relation_name",
                "relation_type",
                "relation",
                "tail_id",
                "tail_type",
                "tail_name",
            ],
        )
        writer.writeheader()

        for row in reader:
            stats.edge_rows_read += 1
            if len(row) != 3:
                stats.bad_edge_rows += 1
                continue

            head_raw, relation_raw, tail_raw = row
            head_id, head_type, head_name = parse_entity(head_raw)
            tail_id, tail_type, tail_name = parse_entity(tail_raw)
            relation = relation_raw.strip()
            relation_type = sanitize_relation_type(relation)
            head_name_value = build_entity_name(
                head_type, head_name, head_id, gene_symbol_map, compound_name_map
            )
            tail_name_value = build_entity_name(
                tail_type, tail_name, tail_id, gene_symbol_map, compound_name_map
            )

            if not head_id or not relation or not tail_id:
                stats.bad_edge_rows += 1
                continue

            writer.writerow(
                {
                    "head_id": head_id,
                    "head_type": head_type,
                    "head_name": head_name_value,
                    "relation_name": relation,
                    "relation_type": relation_type,
                    "relation": relation,
                    "tail_id": tail_id,
                    "tail_type": tail_type,
                    "tail_name": tail_name_value,
                }
            )
            stats.edge_rows_written += 1

            if head_id not in nodes_by_id:
                nodes_by_id[head_id] = (
                    head_type,
                    build_base_name(head_type, head_name, head_id),
                    build_clean_id(head_id),
                    build_gene_symbol(head_type, head_name, gene_symbol_map),
                    safe_label(head_type),
                    build_entity_name(
                        head_type, head_name, head_id, gene_symbol_map, compound_name_map
                    ),
                )
            if tail_id not in nodes_by_id:
                nodes_by_id[tail_id] = (
                    tail_type,
                    build_base_name(tail_type, tail_name, tail_id),
                    build_clean_id(tail_id),
                    build_gene_symbol(tail_type, tail_name, gene_symbol_map),
                    safe_label(tail_type),
                    build_entity_name(
                        tail_type, tail_name, tail_id, gene_symbol_map, compound_name_map
                    ),
                )

    with nodes_csv_path.open("w", encoding="utf-8", newline="") as nodes_handle:
        writer = csv.DictWriter(
            nodes_handle,
            fieldnames=["id", "clean_id", "type", "name", "mapped_name", "symbol", "type_label"],
        )
        writer.writeheader()
        for node_id, (node_type, node_name, clean_id, symbol, type_label, mapped_name) in nodes_by_id.items():
            writer.writerow(
                {
                    "id": node_id,
                    "clean_id": clean_id,
                    "type": node_type,
                    "name": node_name,
                    "mapped_name": mapped_name,
                    "symbol": symbol,
                    "type_label": type_label,
                }
            )
            stats.node_rows_written += 1

    with relations_csv_path.open("w", encoding="utf-8", newline="") as relation_handle:
        writer = csv.DictWriter(
            relation_handle,
            fieldnames=[
                "relation_name",
                "data_source",
                "connected_entity_types",
                "interaction_type",
                "description",
                "reference",
            ],
        )
        writer.writeheader()
        for relation_name, row in relation_info.items():
            writer.writerow({"relation_name": relation_name, **row})
            stats.relation_rows_written += 1

    with entity_sources_csv_path.open("w", encoding="utf-8", newline="") as sources_handle:
        writer = csv.DictWriter(sources_handle, fieldnames=["entity_id", "sources", "source_systems"])
        writer.writeheader()
        for entity_id, sources in entity_sources.items():
            if entity_id in nodes_by_id:
                writer.writerow(
                    {
                        "entity_id": entity_id,
                        "sources": sources,
                        "source_systems": build_source_systems(sources),
                    }
                )
                stats.source_rows_written += 1

    return stats


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert DRKG TSV files to Neo4j-ready CSV files.")
    parser.add_argument(
        "--drkg-tsv",
        type=Path,
        default=Path("drkg/drkg.tsv"),
        help="Path to drkg.tsv file.",
    )
    parser.add_argument(
        "--entity2src-tsv",
        type=Path,
        default=Path("drkg/entity2src.tsv"),
        help="Path to entity2src.tsv file.",
    )
    parser.add_argument(
        "--relation-glossary-tsv",
        type=Path,
        default=Path("drkg/relation_glossary.tsv"),
        help="Path to relation_glossary.tsv file.",
    )
    parser.add_argument(
        "--drugbank-vocab-csv",
        type=Path,
        default=Path("drugbank vocabulary.csv"),
        help="Path to DrugBank vocabulary CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/graph/import_csv"),
        help="Directory where generated CSV files will be stored.",
    )
    parser.add_argument(
        "--no-resolve-gene-names",
        action="store_true",
        help="Disable gene number to symbol mapping via mygene/biopython.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    stats = convert_drkg_tsv_to_csv(
        drkg_tsv_path=args.drkg_tsv,
        entity2src_path=args.entity2src_tsv,
        relation_glossary_path=args.relation_glossary_tsv,
        drugbank_vocab_path=args.drugbank_vocab_csv,
        output_dir=args.output_dir,
        resolve_gene_names=not args.no_resolve_gene_names)

    print("DRKG TSV to CSV conversion completed.")
    print(f"Edge rows read : {stats.edge_rows_read}")
    print(f"Edge rows written: {stats.edge_rows_written}")
    print(f"Bad edge rows : {stats.bad_edge_rows}")
    print(f"Node rows written : {stats.node_rows_written}")
    print(f"Relation rows written: {stats.relation_rows_written}")
    print(f"Entity source rows: {stats.source_rows_written}")


if __name__ == "__main__":
    main()
    

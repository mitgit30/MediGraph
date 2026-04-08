import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


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


def convert_drkg_tsv_to_csv(
    drkg_tsv_path: Path,
    entity2src_path: Path,
    relation_glossary_path: Path,
    output_dir: Path,
) -> ParseStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_csv_path = output_dir / "edges.csv"
    nodes_csv_path = output_dir / "nodes.csv"
    relations_csv_path = output_dir / "relations.csv"
    entity_sources_csv_path = output_dir / "entity_sources.csv"

    stats = ParseStats()
    entity_sources = read_entity_sources(entity2src_path)
    relation_info = read_relation_glossary(relation_glossary_path)

    nodes_by_id: dict[str, tuple[str, str]] = {}

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

            if not head_id or not relation or not tail_id:
                stats.bad_edge_rows += 1
                continue

            writer.writerow(
                {
                    "head_id": head_id,
                    "head_type": head_type,
                    "head_name": head_name,
                    "relation": relation,
                    "tail_id": tail_id,
                    "tail_type": tail_type,
                    "tail_name": tail_name,
                }
            )
            stats.edge_rows_written += 1

            if head_id not in nodes_by_id:
                nodes_by_id[head_id] = (head_type, head_name)
            if tail_id not in nodes_by_id:
                nodes_by_id[tail_id] = (tail_type, tail_name)

    with nodes_csv_path.open("w", encoding="utf-8", newline="") as nodes_handle:
        writer = csv.DictWriter(nodes_handle, fieldnames=["id", "type", "name"])
        writer.writeheader()
        for node_id, (node_type, node_name) in nodes_by_id.items():
            writer.writerow({"id": node_id, "type": node_type, "name": node_name})
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
        writer = csv.DictWriter(sources_handle, fieldnames=["entity_id", "sources"])
        writer.writeheader()
        for entity_id, sources in entity_sources.items():
            if entity_id in nodes_by_id:
                writer.writerow({"entity_id": entity_id, "sources": sources})
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/graph/import_csv"),
        help="Directory where generated CSV files will be stored.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    stats = convert_drkg_tsv_to_csv(
        drkg_tsv_path=args.drkg_tsv,
        entity2src_path=args.entity2src_tsv,
        relation_glossary_path=args.relation_glossary_tsv,
        output_dir=args.output_dir,
    )

    print("DRKG TSV to CSV conversion completed.")
    print(f"Edge rows read : {stats.edge_rows_read}")
    print(f"Edge rows written: {stats.edge_rows_written}")
    print(f"Bad edge rows : {stats.bad_edge_rows}")
    print(f"Node rows written : {stats.node_rows_written}")
    print(f"Relation rows written: {stats.relation_rows_written}")
    print(f"Entity source rows: {stats.source_rows_written}")


if __name__ == "__main__":
    main()

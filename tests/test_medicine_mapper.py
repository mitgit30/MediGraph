from pathlib import Path

from src.graph.retrieval.medicine_mapper import MedicineMapper


def test_medicine_mapper_expands_brand_to_generic():
    mapper = MedicineMapper(
        [
            Path("data/Training/training_labels.csv"),
            Path("data/Validation/validation_labels.csv"),
            Path("data/Testing/testing_labels.csv"),
        ]
    )
    expanded = mapper.expand_terms(["Aceta", "tablet"], max_terms=20)

    print(f"expanded terms: {expanded}")
    assert "Paracetamol" in expanded
test_medicine_mapper_expands_brand_to_generic()
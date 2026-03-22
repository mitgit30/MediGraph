from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    model_name: str = "microsoft/trocr-base-handwritten"
    train_image_dir: str = "data/Training/training_words"
    train_labels_path: str = "data/Training/training_labels.csv"
    val_image_dir: str = "data/Validation/validation_words"
    val_labels_path: str = "data/Validation/validation_labels.csv"
    output_dir: str = "artifacts/trocr_finetuned"
    label_column: str = "MEDICINE_NAME"
    
    max_target_length: int = 64
    batch_size: int = 8
    num_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_workers: int = 0
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(default_factory=lambda: ["query", "value"])


# backward compatibility alias
TrainingPipeline = TrainingConfig

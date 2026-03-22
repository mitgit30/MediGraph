import csv
import math
import random
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_scheduler,
)

from logger.logger import get_logger
from src.config import TrainingConfig

LOGGER = get_logger()


class TrOCRDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        labels_path: str,
        processor: TrOCRProcessor,
        label_column: str,
        max_target_length: int,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.labels_path = Path(labels_path)
        self.processor = processor
        self.label_column = label_column
        self.max_target_length = max_target_length
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(
                f"No valid samples found for image_dir='{self.image_dir}' "
                f"and labels='{self.labels_path}'."
            )

    def _load_samples(self) -> List[Dict[str, str]]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        samples: List[Dict[str, str]] = []
        missing_images = 0
        with self.labels_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_cols = {"IMAGE", self.label_column}
            missing_cols = required_cols - set(reader.fieldnames or [])
            if missing_cols:
                raise ValueError(
                    f"Missing columns {missing_cols} in labels file: {self.labels_path}"
                )

            for row in reader:
                image_name = str(row["IMAGE"]).strip()
                text = str(row[self.label_column]).strip()
                image_path = self.image_dir / image_name

                if not image_name or not text:
                    continue
                if not image_path.exists():
                    missing_images += 1
                    continue

                samples.append({"image_path": str(image_path), "text": text})

        if missing_images:
            LOGGER.warning(
                "Skipped %d rows from %s due to missing image files.",
                missing_images,
                self.labels_path,
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]

        tokenized = self.processor.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids[0]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_compute_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported qlora_compute_dtype '{dtype_name}'. "
            f"Use one of: {sorted(mapping)}."
        )
    return mapping[dtype_name]


def _load_model_with_qlora(
    model_name: str,
    config: TrainingConfig,
    device: torch.device,
) -> VisionEncoderDecoderModel:
    if not config.use_qlora:
        LOGGER.info("QLoRA disabled. Running full fine-tuning.")
        return VisionEncoderDecoderModel.from_pretrained(model_name)

    if device.type != "cuda":
        raise RuntimeError(
            "QLoRA requires CUDA (4-bit bitsandbytes). "
            "Set use_qlora=False for CPU training."
        )

    compute_dtype = _resolve_compute_dtype(config.qlora_compute_dtype)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.qlora_4bit_quant_type,
        bnb_4bit_use_double_quant=config.qlora_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    qlora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.qlora_r,
        lora_alpha=config.qlora_alpha,
        lora_dropout=config.qlora_dropout,
        bias="none",
        target_modules=config.qlora_target_modules,
    )
    model = get_peft_model(model, qlora_config)
    model.print_trainable_parameters()
    LOGGER.info(
        "QLoRA enabled with r=%d alpha=%d dropout=%.2f target_modules=%s",
        config.qlora_r,
        config.qlora_alpha,
        config.qlora_dropout,
        config.qlora_target_modules,
    )
    return model


def _train_one_epoch(
    model: VisionEncoderDecoderModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def _validate(
    model: VisionEncoderDecoderModel,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()

    return total_loss / max(len(dataloader), 1)


def run_training_pipeline(config: TrainingConfig) -> None:
    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    processor = TrOCRProcessor.from_pretrained(config.model_name)
    model = _load_model_with_qlora(config.model_name, config, device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    train_dataset = TrOCRDataset(
        image_dir=config.train_image_dir,
        labels_path=config.train_labels_path,
        processor=processor,
        label_column=config.label_column,
        max_target_length=config.max_target_length,
    )
    val_dataset = TrOCRDataset(
        image_dir=config.val_image_dir,
        labels_path=config.val_labels_path,
        processor=processor,
        label_column=config.label_column,
        max_target_length=config.max_target_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate_fn,
    )

    total_train_steps = len(train_loader) * config.num_epochs
    if total_train_steps == 0:
        raise ValueError("Training steps resolved to zero. Check data and batch size.")

    warmup_steps = math.floor(total_train_steps * config.warmup_ratio)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    output_dir = Path(config.output_dir)
    best_dir = output_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.use_qlora:
        model.to(device)

    best_val_loss = float("inf")
    for epoch in range(1, config.num_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = _validate(model, val_loader, device)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f",
            epoch,
            config.num_epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            LOGGER.info("Saved new best checkpoint to %s", best_dir)

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    LOGGER.info("Training complete. Final checkpoint saved to %s", final_dir)


if __name__ == "__main__":
    run_training_pipeline(TrainingConfig())

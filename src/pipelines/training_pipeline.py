import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import kagglehub
import shutil

from model.config import device, BATCH_SIZE, SAVE_DIR_BEST, SAVE_DIR_FINAL
from model.definition import model, processor
from model.training import train
from model.evaluation import cer_metric, wer_metric

torch.manual_seed(42)
np.random.seed(42)

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")

path = kagglehub.dataset_download("mamun1113/doctors-handwritten-prescription-bd-dataset")
BASE_ROOT = path
BASE_PATH = os.path.join(BASE_ROOT, os.listdir(BASE_ROOT)[0])

print(f"Base path : {BASE_PATH}")
print(f"Contents  : {os.listdir(BASE_PATH)}")

def load_split(split_name):
    split_path = os.path.join(BASE_PATH, split_name)

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    print(f"\nLoading split: {split_name}")

    label_files = [f for f in os.listdir(split_path) if f.lower().endswith(('.csv', '.xlsx'))]

    if not label_files:
        raise FileNotFoundError(f"No label file in {split_path}")

    label_path = os.path.join(split_path, label_files[0])
    df = pd.read_excel(label_path) if label_path.endswith(".xlsx") else pd.read_csv(label_path)

    print(f"Loaded {len(df)} rows | Columns: {df.columns.tolist()}")

    image_col, text_col = None, None
    for col in df.columns:
        cl = str(col).lower()
        if cl in ['image_name', 'filename', 'img', 'image']: image_col = col
        if cl in ['word', 'label', 'text', 'transcription']: text_col = col

    image_col = image_col or df.columns[0]
    text_col  = text_col  or df.columns[1]

    df = df.rename(columns={image_col: "file_name", text_col: "text"})

    subfolder_map = {
        "Training": "training_words",
        "Validation": "validation_words",
        "Testing": "testing_words"
    }

    image_folder = os.path.join(split_path, subfolder_map.get(split_name, "words"))

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    df["file_name"] = df["file_name"].apply(
        lambda x: os.path.join(image_folder, x) if isinstance(x, str) else None
    )

    df["exists"] = df["file_name"].apply(lambda x: os.path.exists(x) if x else False)

    dropped = (~df["exists"]).sum()
    if dropped:
        print(f"Dropping {dropped} missing images")

    df = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.strip()

    print(f"Usable samples: {len(df)}")

    return Dataset.from_pandas(df)


train_ds = load_split("Training")
val_ds   = load_split("Validation")
test_ds  = load_split("Testing")

print(f"\nSummary → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

def load_image(example):
    try:
        example["image"] = Image.open(example["file_name"]).convert("RGB")
    except Exception:
        example["image"] = None
    return example

train_ds = train_ds.map(load_image, num_proc=1)
val_ds   = val_ds.map(load_image,   num_proc=1)
test_ds  = test_ds.map(load_image,  num_proc=1)

train_ds = train_ds.filter(lambda x: x["image"] is not None)
val_ds   = val_ds.filter(lambda x: x["image"] is not None)
test_ds  = test_ds.filter(lambda x: x["image"] is not None)

print(f"After filtering --> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

def collate_fn(batch):
    images = [ex["image"] for ex in batch]
    texts  = [ex["text"]  for ex in batch]

    encoding = processor(
        images=images,
        text=texts,
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt"
    )

    labels = encoding["labels"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": encoding["pixel_values"],
        "labels": labels
    }

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

optimizer, scheduler, scaler, history, best_cer = train(
    model, processor, train_loader, val_loader, device
)

print("Starting full fine-tune (encoder + decoder)...\n")

for epoch in range(10):

    model.train()
    train_loss_accum = 0.0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10 [Train]")

    for step, batch in enumerate(pbar):

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / 4

        scaler.scale(loss).backward()
        train_loss_accum += loss.item() * 4

        if (step + 1) % 4 == 0:

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        pbar.set_postfix({"train_loss": f"{train_loss_accum / (step + 1):.4f}"})

    avg_train_loss = train_loss_accum / len(train_loader)

    model.eval()
    val_loss_accum = 0.0
    all_preds, all_labels_str = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/10 [Val]"):

            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)

            val_loss_accum += outputs.loss.item()

            generated_ids = model.generate(pixel_values)
            pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)

            label_ids = batch["labels"].clone()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

            all_preds.extend(pred_str)
            all_labels_str.extend(label_str)

    avg_val_loss = val_loss_accum / len(val_loader)

    cer = cer_metric.compute(predictions=all_preds, references=all_labels_str)
    wer = wer_metric.compute(predictions=all_preds, references=all_labels_str)

    enc_lr = optimizer.param_groups[0]["lr"]
    dec_lr = optimizer.param_groups[1]["lr"]

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["cer"].append(cer)
    history["wer"].append(wer)
    history["enc_lr"].append(enc_lr)
    history["dec_lr"].append(dec_lr)

    print(f"\nEpoch {epoch+1:>2} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"CER: {cer:.4f} | "
          f"WER: {wer:.4f} | "
          f"Enc LR: {enc_lr:.2e} | "
          f"Dec LR: {dec_lr:.2e}")

    if cer < best_cer:
        best_cer = cer
        model.save_pretrained(SAVE_DIR_BEST)
        processor.save_pretrained(SAVE_DIR_BEST)
        print(f"  Best model saved (CER: {cer:.4f})")

model.save_pretrained(SAVE_DIR_FINAL)
processor.save_pretrained(SAVE_DIR_FINAL)

print(f"\nTraining complete. Best CER: {best_cer:.4f}")
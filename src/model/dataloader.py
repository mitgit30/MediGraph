from torch.utils.data import DataLoader

def collate_fn(batch, processor):

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


def get_dataloaders(train_ds, val_ds, batch_size, processor):

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader
import os
import pandas as pd
from PIL import Image
from datasets import Dataset

def load_split(BASE_PATH, split_name):

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
        if cl in ['image_name', 'filename', 'img', 'image']:
            image_col = col
        if cl in ['word', 'label', 'text', 'transcription']:
            text_col = col

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


def load_image(example):

    try:
        example["image"] = Image.open(example["file_name"]).convert("RGB")
    except Exception:
        example["image"] = None

    return example


def apply_preprocessing(train_ds, val_ds, test_ds):

    train_ds = train_ds.map(load_image, num_proc=1)
    val_ds   = val_ds.map(load_image,   num_proc=1)
    test_ds  = test_ds.map(load_image,  num_proc=1)

    train_ds = train_ds.filter(lambda x: x["image"] is not None)
    val_ds   = val_ds.filter(lambda x: x["image"] is not None)
    test_ds  = test_ds.filter(lambda x: x["image"] is not None)

    return train_ds, val_ds, test_ds
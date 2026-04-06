from torch.optim.lr_scheduler import OneCycleLR
import torch
from tqdm import tqdm

NUM_EPOCHS = 10
GRADIENT_ACCUMULATION = 4

def train(model, processor, train_loader, val_loader, device):

    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 5e-6},
        {"params": decoder_params, "lr": 1e-4},
    ], weight_decay=0.01)

    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION) * NUM_EPOCHS

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[5e-6, 1e-4],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos"
    )

    scaler = torch.amp.GradScaler("cuda")

    best_cer = float("inf")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "cer": [],
        "wer": [],
        "enc_lr": [],
        "dec_lr": [],
    }

    return optimizer, scheduler, scaler, history, best_cer
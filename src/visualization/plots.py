import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_loss(history):

    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="s", linestyle="--", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig("chart_loss.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_cer_wer(history):

    epochs = history["epoch"]

    cer_pct = [c * 100 for c in history["cer"]]
    wer_pct = [w * 100 for w in history["wer"]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, cer_pct, marker="o", label="CER %")
    plt.plot(epochs, wer_pct, marker="s", linestyle="--", label="WER %")

    plt.xlabel("Epoch")
    plt.ylabel("Error (%)")
    plt.title("CER & WER vs Epoch")
    plt.legend()
    plt.grid(True)

    plt.savefig("chart_cer_wer.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_lr(history):

    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["enc_lr"], marker="o", label="Encoder LR")
    plt.plot(epochs, history["dec_lr"], marker="s", linestyle="--", label="Decoder LR")

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)

    plt.savefig("chart_lr.png", dpi=150, bbox_inches="tight")
    plt.show()
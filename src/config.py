import torch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoint
MODEL_CKPT = "microsoft/trocr-base-handwritten"

# Training params
NUM_EPOCHS = 10
GRADIENT_ACCUMULATION = 4
BATCH_SIZE = 8

# Learning rates
ENCODER_LR = 5e-6
DECODER_LR = 1e-4

# Save paths
SAVE_DIR_BEST  = "./trocr-prescription-best"
SAVE_DIR_FINAL = "./trocr-prescription-final"
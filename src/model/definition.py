from transformers import TrOCRProcessor, VisionEncoderDecoderModel, GenerationConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CKPT = "microsoft/trocr-base-handwritten"

processor = TrOCRProcessor.from_pretrained(MODEL_CKPT)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_CKPT)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

model.generation_config = GenerationConfig(
    max_length=32,
    early_stopping=True,
    no_repeat_ngram_size=3,
    length_penalty=2.0,
    num_beams=4,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.sep_token_id,
    decoder_start_token_id=processor.tokenizer.cls_token_id
)

for param in model.parameters():
    param.requires_grad = True

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)")
print(f"Device   : {device}")
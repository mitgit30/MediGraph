from PIL import Image as PILImage
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_model(save_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = TrOCRProcessor.from_pretrained(save_dir)
    model = VisionEncoderDecoderModel.from_pretrained(save_dir).to(device)
    model.eval()

    return model, processor, device


def predict(image_path, model, processor, device):

    img = PILImage.open(image_path).convert("RGB")

    pixel_values = processor(
        images=img,
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    print(f"\n{'─'*35}")
    print(f" File : {image_path}")
    print(f" Prediction : {prediction}")
    print(f"{'─'*35}\n")

    img.show()
    
import 
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# Initialize once (on import)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

def caption_image_path(image_path: str, max_length: int = 30) -> str:
    """
    Given a local image file path, returns a short caption string.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=max_length)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        # fallback: empty caption
        return ""
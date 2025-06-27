import torch
import os
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "fastapi-vit-app", "vit_weights_v1_20250627.pth")
IMAGE_PATH = os.path.join(BASE_DIR, "fastapi-vit-app", "sample.jpg")

# Load checkpoint
checkpoint = torch.load(WEIGHTS_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu')
state_dict = checkpoint['model_state_dict']
class_names = checkpoint['class_names']  # Optional, for prediction label

# Initialize model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=len(class_names))
model.load_state_dict(state_dict)
model.eval()

# Load and process image
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

# Print result
print(f"Predicted class: {class_names[predicted_class_idx]}")
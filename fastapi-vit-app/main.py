from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
import io
import json

# Load model config
MODEL_NAME = "google/vit-base-patch16-224-in21k"
NUM_CLASSES = 7  # Change based on your training
MODEL_PATH = "vit_weights_v1_20250627.pth"

# Load processor
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# Load model
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Class names
class_names = checkpoint.get("class_names", [f"Class {i}" for i in range(NUM_CLASSES)])

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_idx = torch.argmax(probs, dim=1).item()
            predicted_class = class_names[predicted_idx]
            confidence = probs[0][predicted_idx].item()
        
        return {
            "class": predicted_class,
            "confidence": round(confidence * 100, 2)
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
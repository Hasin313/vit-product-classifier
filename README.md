🧠 Product Image Classification with ViT (Vision Transformer)
This repository offers a powerful and scalable deep learning solution for automated product image classification using a pretrained Vision Transformer (ViT) model. The system is designed to classify images into 7 product categories, such as bags, watches, sunglasses, and more, based on their visual features. It combines state-of-the-art machine learning techniques with modern web deployment frameworks to deliver both accuracy and usability.

🔍 What This Project Includes
✅ Pretrained Vision Transformer (ViT) Model
Leveraging google/vit-base-patch16-224-in21k, fine-tuned on over 41,000 labeled product images for robust feature extraction and classification.

✅ FastAPI Backend for API Inference
A production-ready, asynchronous Python API using FastAPI. Upload any image via a REST API endpoint and get instant category predictions.

✅ Gradio Web Interface (No-Code UI)
An intuitive drag-and-drop browser interface that allows users to upload an image and instantly view predictions with confidence scores — no code required.

✅ Training + Evaluation Notebooks
Includes Jupyter notebooks for full training pipeline, preprocessing, model evaluation, and metrics visualization using PyTorch, HuggingFace Transformers, and Albumentations.

✅ Google Drive Hosted Weights (~327MB)
Pretrained model weights are hosted on Google Drive for easy download and integration.

This project is ideal for:

🛒 E-commerce platforms looking to automate product tagging and visual categorization

🔬 Researchers and students exploring computer vision applications using transformer architectures

🧑‍💻 Developers who want a ready-to-use image classification model with clean deployment interfaces

🧪 Educators demonstrating real-world machine learning deployment pipelines
---

## 📁 Project Structure

```
.
├── data/                          # Training and evaluation datasets (ignored)
├── fastapi-vit-app/              # FastAPI application
│   ├── main.py                   # FastAPI app entrypoint
│   ├── sample.jpg                # Sample image for testing
│   ├── class_map.json            # JSON file for label mapping
│   └── vit_weights_v1_20250627.pth   # 🔽 Downloaded model weights (see below)
├── gradio_app.py                 # Gradio web interface
├── test_model.py                 # Script to test model on a local image
├── utils.ipynb                   # Utility notebook
├── vit-model.ipynb               # Model training notebook
├── requirements.txt              # Required Python libraries
├── README.md                     # Project guide
└── .gitignore                    # Ignored files
```

---

## 🔧 Setup Instructions

### 1. 📦 Create Virtual Environment & Install Requirements

```bash
python -m venv venv
# Activate environment
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate

# Upgrade pip and install packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. 🔽 Download Pretrained Model Weights

This model was trained on 41,000+ product images using the Vision Transformer architecture `google/vit-base-patch16-224-in21k`.

To run both the FastAPI backend and Gradio UI, you must first download the trained model weights and save them inside the `fastapi-vit-app/` directory, like this:

```
project_root/
├── fastapi-vit-app/
│   └── vit_weights_v1_20250627.pth  ← Save the model here
├── gradio_app.py
├── test_model.py
└── ...
```

📌 **Why this location?**  
Both the FastAPI app (`main.py`) and the Gradio interface (`gradio_app.py`) are pre-configured to look for the weights file inside the `fastapi-vit-app/` directory. Placing the file elsewhere will result in errors unless you manually change the code paths.

✅ **Steps to download:**

Make sure you have the `gdown` tool installed:

```bash
pip install gdown
```

Then run the command to download weights (size:- 327MB):

```bash
gdown --id 1MLtH8djhOFUZ32QLBZIYSufSCTk9II-t -O fastapi-vit-app/vit_weights_v1_20250627.pth
```

Once downloaded, you're all set to run the prediction scripts, Gradio interface, or FastAPI server without needing to change any paths.

---

## 🚀 Run Inference

### 1. 🧪 Test Prediction Locally

To test the model on a sample image:

```bash
python test_model.py
```

---

## 🌐 Gradio Web App (No-Code UI)

Let users upload an image and see prediction results with confidence scores.

### ▶️ Run the App

```bash
python gradio_app.py
```

Then open your browser at:

```
http://127.0.0.1:7860/
```

You’ll see an upload interface. Drop any image of a product (e.g., sunglasses, bag) to get results.

---

## ⚡ FastAPI Inference Backend (API)

This provides a backend for programmatic prediction using `/predict`.

### ▶️ Run the FastAPI App

```bash
cd fastapi-vit-app
uvicorn main:app --reload
```

Then open the API docs at:

```
http://127.0.0.1:8000/docs
```

You can upload images here using Swagger UI.

---

### 🧪 Example curl Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample.jpg;type=image/jpeg'
```

---

## 📚 Model Info

- Architecture: ViT-base-patch16-224-in21k  
- Training Accuracy: 99.99%  
- Validation Accuracy: 99.82%  
- Test Accuracy: 99.65%  
- Total Parameters: 85M+  
- Input size: 224x224  
- Best performance on: glasses  
- Most challenging class: bags  

---

## 🙌 Acknowledgements

- HuggingFace Transformers  
- PyTorch  
- Gradio & FastAPI for easy deployment  

---

## 🔐 License

This project is for research and educational use only. Not for commercial distribution without permission.
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import gradio as gr
import os

# ======== Load Model Weights ========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "fastapi-vit-app", "vit_weights_v1_20250627.pth")
checkpoint = torch.load(WEIGHTS_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Get class names and other info
class_names = checkpoint['class_names']
label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']

# Load model architecture and weights
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load image processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# ======== Enhanced Prediction Function ========
def predict(image):
    if image is None:
        return "🤖 Please upload an image to get started!"
    
    try:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, pred_label_idx = torch.max(probs, dim=1)
            predicted_class = idx_to_label[pred_label_idx.item()]
            
            # Enhanced output with emojis and formatting
            confidence_score = confidence.item()
            if confidence_score > 0.8:
                emoji = "🎯"
                confidence_text = "Very High"
            elif confidence_score > 0.6:
                emoji = "✅"
                confidence_text = "High"
            elif confidence_score > 0.4:
                emoji = "⚡"
                confidence_text = "Medium"
            else:
                emoji = "🤔"
                confidence_text = "Low"
            
            return f"{emoji} **Prediction: {predicted_class}**\n\n📊 Confidence: {confidence_text} ({confidence_score:.2%})"
    
    except Exception as e:
        return f"❌ Error processing image: {str(e)}"

# ======== Custom CSS Styling ========
custom_css = """
/* Main container styling - Dark theme */
.gradio-container {
    background: #1a1a1a !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    color: #e0e0e0 !important;
}

/* Header styling */
.gradio-container h1 {
    text-align: center !important;
    color: #ffffff !important;
    font-size: 2.5em !important;
    font-weight: bold !important;
    margin-bottom: 10px !important;
}

/* Description styling */
.gradio-container p {
    text-align: center !important;
    color: #b0b0b0 !important;
    font-size: 1.2em !important;
    margin-bottom: 30px !important;
}

/* Input/Output containers */
.input-container, .output-container {
    background: #2d2d2d !important;
    border-radius: 15px !important;
    padding: 25px !important;
    margin: 15px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    border: 1px solid #404040 !important;
}

/* Image upload area */
.image-upload {
    border: 3px dashed #6c5ce7 !important;
    border-radius: 15px !important;
    background: #333333 !important;
    min-height: 200px !important;
}

.image-upload:hover {
    border-color: #a29bfe !important;
    background: #3a3a3a !important;
}

/* Button styling */
.btn-primary {
    background: #6c5ce7 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 30px !important;
    font-weight: bold !important;
    color: white !important;
    font-size: 1.1em !important;
}

.btn-primary:hover {
    background: #5a4fcf !important;
}

/* Output text styling */
.output-text {
    background: #363636 !important;
    color: #ffffff !important;
    padding: 20px !important;
    border-radius: 12px !important;
    font-size: 1.3em !important;
    font-weight: 500 !important;
    text-align: center !important;
    border: 1px solid #505050 !important;
}

/* Markdown text styling */
.gradio-container .markdown {
    color: #e0e0e0 !important;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .gradio-container h1 {
        font-size: 2em !important;
    }
    .input-container, .output-container {
        margin: 10px 5px !important;
        padding: 15px !important;
    }
}
"""

# ======== Launch Enhanced Gradio App ========
with gr.Blocks(
    css=custom_css,
    title="🎨 AI Product Classifier",
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="slate",
    ).set(
        body_background_fill="#1a1a1a",
        body_text_color="#e0e0e0",
        block_background_fill="#2d2d2d",
        block_border_color="#404040",
        input_background_fill="#333333",
        button_primary_background_fill="#6c5ce7",
        button_primary_text_color="#ffffff"
    )
) as demo:
    
    # Header
    gr.Markdown(
        """
        # 🚀 AI-Powered Product Classification
        ### Upload any product image and watch our Vision Transformer work its magic! ✨
        """,
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Upload Your Image")
            image_input = gr.Image(
                type="pil",
                label="Drop your product image here",
                elem_classes=["image-upload"]
            )
            
            predict_btn = gr.Button(
                "🔮 Classify Product",
                variant="primary",
                elem_classes=["btn-primary"]
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Classification Result")
            output_text = gr.Markdown(
                "🤖 Ready to analyze your product image!",
                elem_classes=["output-text"]
            )
    

    # Info section
    with gr.Accordion("ℹ️ How It Works", open=False):
        gr.Markdown(
            """
            This app uses a **Vision Transformer (ViT)** model to classify product images:
            
            🧠 **Model**: Google's ViT-Base fine-tuned for product classification  
            ⚡ **Speed**: Lightning-fast predictions in seconds  
            🎯 **Accuracy**: High-confidence predictions with probability scores  
            🌈 **Classes**: Supports multiple product categories  
            
            Simply upload an image and click the classify button to see the magic happen! ✨
            """
        )
    
    # Event handlers
    predict_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[output_text]
    )
    
    # Auto-predict on image upload
    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=[output_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your model (using the path from your project)
# compile=False avoids version mismatch errors with optimizers
model = load_model("deepfake_model.h5", compile=False)

def predict_deepfake(input_img):
    if input_img is None:
        return "Please upload an image.", 0
    
    # 1. Preprocessing (Matching your notebook logic)
    # Gradio passes the image as a RGB numpy array
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)
    
    # 2. Prediction
    pred = model.predict(img)[0][0]
    
    # 3. Logic based on your working notebook (> 0.5 = Real)
    if pred > 0.5:
        label = "REAL ✅"
        confidence = float(pred)
    else:
        label = "FAKE ❌"
        confidence = float(1 - pred)
        
    return label, f"{confidence:.2%}"

# Create the Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Deepfake Detection Forensic Tool")
    gr.Markdown("Upload an image to analyze for AI-generated inconsistencies.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Input Evidence")
            run_button = gr.Button("Analyze Image", variant="primary")
        
        with gr.Column():
            label_output = gr.Textbox(label="Result Classification")
            conf_output = gr.Textbox(label="System Confidence")

    # Connect the logic
    run_button.click(
        fn=predict_deepfake, 
        inputs=image_input, 
        outputs=[label_output, conf_output]
    )

    gr.Examples(
        examples=["test.png"], # You can add paths to your test images here
        inputs=image_input
    )

if __name__ == "__main__":
    demo.launch()
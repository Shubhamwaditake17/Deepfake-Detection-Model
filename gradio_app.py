import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr
import cv2
import numpy as np
import tensorflow as tf

# 1. THE "DTYPE" FIX
# This dummy class prevents the 'Unknown dtype policy' error
class DTypePolicy:
    def __init__(self, name="float32", **kwargs):
        self.name = name
    def get_config(self):
        return {'name': self.name}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# 2. THE "INPUT" FIX
class UniversalInput(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        for key in ['batch_shape', 'batch_input_shape', 'shape', 'ragged', 'sparse']:
            kwargs.pop(key, None)
        super().__init__(**kwargs)

# Register these fixes
custom_objs = {'DTypePolicy': DTypePolicy, 'InputLayer': UniversalInput}

print("🔍 Attempting final forensic load...")
model = None
try:
    # First attempt: Loading with patches
    model = tf.keras.models.load_model("deepfake_model.h5", custom_objects=custom_objs, compile=False)
    print("✅ Success! Model loaded with custom patches.")
except Exception as e:
    print(f"⚠️ Primary load failed, trying legacy bridge...")
    try:
        # Second attempt: Using the tf-keras legacy library
        import tf_keras
        model = tf_keras.models.load_model("deepfake_model.h5", compile=False)
        print("✅ Success! Model loaded via tf-keras.")
    except Exception as e2:
        print(f"❌ Critical Failure: {e2}")

def predict_deepfake(input_img):
    if model is None: return "ERROR: Model not loaded", "0%"
    try:
        # Preprocessing (BGR + Float32 for M1 GPU stability)
        img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR) 
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        pred = model.predict(img, verbose=0)[0][0]
        
        if pred > 0.5:
            return "REAL ✅", f"{float(pred):.2%}"
        else:
            return "FAKE ❌", f"{float(1 - pred):.2%}"
    except Exception as err:
        return f"Prediction Error: {str(err)}", "Check Terminal"

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🛡️ CyberScan: Deepfake Detection Forensic Tool")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Input Evidence")
            btn = gr.Button("RUN ANALYSIS", variant="primary")
        with gr.Column():
            lbl_out = gr.Textbox(label="Result Classification")
            conf_out = gr.Textbox(label="Confidence Level")
    
    btn.click(fn=predict_deepfake, inputs=img_in, outputs=[lbl_out, conf_out])

if __name__ == "__main__":
    demo.launch()

import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("model/deepfake_model.h5", compile=False)

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = img.reshape(1, 128, 128, 3)

    pred = model.predict(img)[0][0]

    if pred > 0.7:
        print("Real ✅")
    else:
        print("Fake ❌")

    print(f"Confidence: {pred:.4f}")

# Test
predict_image("test.jpg")  # put any test image

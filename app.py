import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ---------------------------
# ENVIRONMENT SETTINGS
# ---------------------------
# Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Force CPU (if using Render or local CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ---------------------------
# LOAD MODEL
# ---------------------------
MODEL_PATH = "./Brain_tumor_XceptionModel.h5"  # Update this if your path is different
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ---------------------------
# IMAGE SIZE AND CLASS LABELS
# ---------------------------
IMG_SIZE = (224, 224)  # Must match your model's input size
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Update if needed

# ---------------------------
# STREAMLIT APP LAYOUT
# ---------------------------
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image and get predictions from the Xception model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load image and convert to RGB
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Predict
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        pred_label = class_labels[pred_class]
        confidence = float(np.max(preds)) * 100

        # Display results
        st.success(f"Prediction: {pred_label}")
        st.info(f"Confidence: {round(confidence, 2)}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

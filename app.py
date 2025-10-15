import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# ENVIRONMENT SETTINGS
# ---------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ---------------------------
# LOAD MODEL
# ---------------------------
MODEL_PATH = "./Brain_tumor_XceptionModel.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ---------------------------
# IMAGE SIZE AND CLASS LABELS
# ---------------------------
IMG_SIZE = (224, 224)
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ---------------------------
# DATA STORAGE
# ---------------------------
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=['Class', 'Confidence'])

# ---------------------------
# STREAMLIT DASHBOARD LAYOUT
# ---------------------------
st.set_page_config(page_title="Brain Tumor Dashboard", layout="wide")
st.title("🧠 Brain Tumor Detector Dashboard")

# Sidebar: Upload Image
st.sidebar.header("Upload MRI Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Main: Prediction & Visualization
col1, col2 = st.columns([1,1])

if uploaded_file is not None:
    try:
        # Load & display image
        img = Image.open(uploaded_file).convert('RGB')
        col1.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess
        img_array = np.array(img.resize(IMG_SIZE))
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        preds = model.predict(img_array)
        pred_class_idx = np.argmax(preds, axis=1)[0]
        pred_label = class_labels[pred_class_idx]
        confidence = float(np.max(preds)) * 100

        # Display prediction
        col2.success(f"Prediction: {pred_label}")
        col2.info(f"Confidence: {round(confidence, 2)}%")

        # Save to history
        st.session_state['history'] = pd.concat(
            [st.session_state['history'], 
             pd.DataFrame({'Class':[pred_label], 'Confidence':[confidence]})],
            ignore_index=True
        )

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ---------------------------
# Dashboard Charts
# ---------------------------
if not st.session_state['history'].empty:
    st.subheader("📊 Prediction History")
    
    # Show data table
    st.dataframe(st.session_state['history'])

    # Class distribution
    st.subheader("Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Class', data=st.session_state['history'], palette="viridis", ax=ax1)
    ax1.set_title("Predicted Classes Count")
    st.pyplot(fig1)

    # Confidence distribution
    st.subheader("Confidence Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(st.session_state['history']['Confidence'], bins=10, kde=True, color='skyblue', ax=ax2)
    ax2.set_title("Prediction Confidence (%)")
    st.pyplot(fig2)

from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the SavedModel folder (replace with your folder path)
MODEL_PATH = "./Brain_tumor_XceptionModel.h5"  # path to the folder
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size (must match training)
IMG_SIZE = (224, 224)

# Define class labels (update according to your training classes)
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        # Save uploaded file
        file_path = os.path.join("static", file.filename)
        os.makedirs("static", exist_ok=True)
        file.save(file_path)

        # Preprocess image
        img = image.load_img(file_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

        # Predict
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        pred_label = class_labels[pred_class]
        confidence = float(np.max(preds)) * 100

        return render_template("result.html",
                               file_path=file_path,
                               prediction=pred_label,
                               confidence=round(confidence, 2))

# Run app
if __name__ == "__main__":
    app.run(debug=True)

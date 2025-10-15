from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ---------------------------
# ENVIRONMENT SETTINGS
# ---------------------------
# Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Force CPU (Render does not provide GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ---------------------------
# INITIALIZE FLASK APP
# ---------------------------
app = Flask(__name__)

# ---------------------------
# LOAD MODEL
# ---------------------------
MODEL_PATH = "./Brain_tumor_XceptionModel.h5"  # Ensure this path matches your project
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# ---------------------------
# IMAGE SIZE AND CLASS LABELS
# ---------------------------
IMG_SIZE = (224, 224)  # Must match training size
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Update if needed

# ---------------------------
# ROUTES
# ---------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        try:
            # Save uploaded file safely
            os.makedirs("static/uploads", exist_ok=True)
            file_path = os.path.join("static/uploads", file.filename)
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
        except Exception as e:
            return f"Error during prediction: {e}"

# ---------------------------
# RUN APP
# ---------------------------
if __name__ == "__main__":
    # Bind to Render's PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

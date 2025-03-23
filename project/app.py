from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained deepfake detection model
model = tf.keras.models.load_model("Xception_model.h5.h5")

# Define the image size expected by your model
IMG_SIZE = (224, 224)  # Change based on your model's input
CLASS_NAMES = ["Real", "Deepfake"]  # Update based on your labels

@app.route('/')
def home():
    return render_template('index.html')  # Load the frontend UI

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure a file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        # Open and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize(IMG_SIZE)  # Resize image to match model input
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Perform prediction
        predictions = model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]  # Get predicted label
        confidence = float(np.max(predictions))  # Get confidence score

        return jsonify({'prediction': predicted_class, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

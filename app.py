from flask import Flask, render_template, request
import pickle
import cv2
import numpy as np
import base64
import logging
import traceback
import io  # Import io for handling bytes

# Configure logging (do this once, preferably at the top of your app.py)
logging.basicConfig(level=logging.ERROR)  # Set logging level
logger = logging.getLogger(__name__)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        image_data = data.get('image') # Handle potential missing 'image' key
        if not image_data:
            logger.error("No image data received from client.")
            return "Error, No Image", 400

        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None: # Check if imdecode failed
            logger.error("cv2.imdecode failed to decode image.")
            return "Error, Invalid Image Format", 400

        logger.info(f"Frame shape: {frame.shape}, Frame type: {frame.dtype}")
        processed_frame = preprocess_frame(frame)
        logger.info(f"Processed frame shape: {processed_frame.shape}, Processed frame type: {processed_frame.dtype}")

        predictions = model.predict(processed_frame)
        class_idx = np.argmax(predictions)
        label = fashion_classes[class_idx]
        probability = predictions[0][class_idx] * 100

        logger.info(f"Predicted Label: {label}, Probability: {probability:.2f}%")
        return f"{label},{round(probability, 2)}"

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in prediction: {e}\n{tb}")
        return "Error,Invalid frame", 500
# Initialize Flask app
app = Flask(__name__)

# Load the model
model_path = "models/fashion_mnist_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define class labels
fashion_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Convert to numpy array and preprocess
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_frame = preprocess_frame(frame)

        # Make prediction
        predictions = model.predict(processed_frame)
        class_idx = np.argmax(predictions)
        label = fashion_classes[class_idx]
        probability = predictions[0][class_idx] * 100

        print(f"Predicted Label: {label}, Probability: {probability:.2f}%")
        return f"{label},{round(probability, 2)}"
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error,Invalid frame", 500

if __name__ == "__main__":
    app.run(debug=False)
    
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

from flask import Flask, render_template, request
import random
import time
import traceback
import logging
import os

# Initialize Flask app
app = Flask(__name__)

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define class labels
fashion_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Define probability ranges for each class
hardcoded_detections = {
    "T-shirt/top": (60, 80),
    "Trouser": (65, 75),
    "Pullover": (70, 90),
    "Dress": (55, 85),
    "Coat": (62, 78),
    "Sandal": (75, 82),
    "Shirt": (68, 88),
    "Sneaker": (58, 72),
    "Bag": (72, 92),
    "Ankle boot": (61, 79),
}

# Variables to manage stable detection logic
current_stable_detection = None
stable_start_time = None
stable_duration = 3  # Stable period in seconds


def generate_random_probability(min_prob, max_prob):
    """Generates a random float probability between min_prob and max_prob."""
    return round(random.uniform(min_prob, max_prob), 2)


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    global current_stable_detection, stable_start_time

    try:
        current_time = time.time()

        if current_stable_detection and (current_time - stable_start_time) < stable_duration:
            # Continue with the current stable detection
            label, (min_prob, max_prob) = current_stable_detection
            probability = generate_random_probability(min_prob, max_prob)
            logger.info(f"Stable Prediction: {label}, Probability: {probability:.2f}%")
            return f"{label},{probability:.2f}"
        else:
            # Switch to a new stable detection after the stable duration
            label = random.choice(list(hardcoded_detections.keys()))
            current_stable_detection = (label, hardcoded_detections[label])
            stable_start_time = current_time
            logger.info(f"Switching to New Stable Detection: {label}")

            # Generate probability for the new detection
            label, (min_prob, max_prob) = current_stable_detection
            probability = generate_random_probability(min_prob, max_prob)
            logger.info(f"Stable Prediction: {label}, Probability: {probability:.2f}%")
            return f"{label},{probability:.2f}"

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in prediction: {e}\n{tb}")
        return "Error,Invalid frame", 500


if __name__ == "__main__":
    # Set port for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from app.utils.preprocess import preprocess_image
import numpy as np
from flask_cors import CORS
from PIL import Image
import io

# Initialize Flask app and load model
app = Flask(__name__)
CORS(app)
model = load_model('app/model/emotion_model.keras')

# Define possible emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Check if the image file was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided. Please send an image file with key 'image'"}), 400

        # Read and process the image
        image_file = request.files['image']
        image = Image.open(image_file)
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        emotion_label = np.argmax(prediction)
        result = emotions[emotion_label]
        confidence = float(prediction[0][emotion_label])
        
        return jsonify({
            'emotion': result,
            'confidence': confidence,
            'predictions': {emotion: float(pred) for emotion, pred in zip(emotions, prediction[0])}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

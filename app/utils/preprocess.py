# app/utils/preprocess.py
import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess image for emotion detection model.
    Accepts a PIL Image object
    """
    try:
        # Convert PIL Image to numpy array
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess image
        img = cv2.resize(img, (48, 48))  # Resize to model input size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = img.astype('float32') / 255.0  # Normalize
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        return img

    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")
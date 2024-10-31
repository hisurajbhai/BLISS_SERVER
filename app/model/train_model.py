import cv2 
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import kagglehub

# Function to download dataset
def download_dataset():
    path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
    print("Path to dataset files:", path)
    return path

# Define the CNN model
def create_emotion_model(input_shape=(48, 48, 1)):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Specify the input shape using Input layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # Assuming 7 emotion classes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load dataset, preprocess, and prepare for training
def load_data(data_dir):
    X = []
    y = []
    label_map = {}  # Create a mapping of labels to integers
    label_id = 0
    
    # Assuming your images are in subdirectories named by class labels
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            label_map[label] = label_id  # Map the label to an integer
            label_id += 1  # Increment the label id for the next class
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                # Load the image, resize it, and normalize pixel values
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue  # Skip if the image couldn't be loaded
                img = cv2.resize(img, (48, 48))  # Resize to fit model input
                X.append(img)
                y.append(label_map[label])  # Use the mapped integer label
    
    # Convert lists to numpy arrays
    X = np.array(X, dtype='float32') / 255.0  # Normalize to [0, 1]
    y = np.array(y, dtype='int')

    # One-hot encode the labels
    y = to_categorical(y, num_classes=len(label_map))  # Adjust num_classes based on the number of unique labels

    return X, y

# Training function
def train_model():
    X, y = load_data(r"C:\Users\prate\Emotion_Detc_Model\emotion_detection_project\data")

    model = create_emotion_model((48, 48, 1))
    model.fit(X, y, epochs=25, validation_split=0.2)  # Use validation split instead of separate test data
    # model.save('app/model/emotion_model.h5')
    model.save('app/model/emotion_model.keras')  # Save in the new Keras format
    print("Model training complete and saved as 'emotion_model.keras'")

if __name__ == "__main__":
    train_model()

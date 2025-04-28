import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
cnn = load_model(r"D:\Pneumonia _final\cnn_pneumonia_model.h5")

# Function to preprocess the image
def preprocess_image(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = cv2.resize(img, target_size)  # Resize to target size (64x64)
    img = img / 255.0  # Normalize pixel values (0 to 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (for model input)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (grayscale)
    return img

# Streamlit UI
st.title("Pneumonia Detection Using CNN")
st.write("Upload a chest X-ray image to classify it as 'Pneumonia Detected' or 'Normal Lungs'.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open("temp_image.jpg", "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    try:
        # Preprocess the uploaded image
        processed_image = preprocess_image("temp_image.jpg")
        st.write("Image preprocessing successful!")
        st.write(f"Processed Image Shape: {processed_image.shape}")

        # Make a prediction
        prediction = cnn.predict(processed_image)

        # Display the result
        if prediction[0][0] > 0.5:
            st.success("Prediction: Pneumonia Detected")
        else:
            st.success("Prediction: Normal Lungs")
    except Exception as e:
        st.error(f"An error occurred: {e}")
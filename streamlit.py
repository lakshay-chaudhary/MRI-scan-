import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

# Load your trained model
model_path = r"C:\Users\lakshay\Desktop\mri scan\nested_unet_partial.keras"
model = load_model(model_path)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess the image for model prediction."""
    img_resized = cv2.resize(image, (256, 256))  # Adjust size as per model input
    img_normalized = img_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_normalized, axis=0)  # Add batch dimension

def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Postprocess the predicted mask to binary (0 and 1)."""
    mask_binary = (mask > 0.5).astype(np.uint8)  # Apply threshold
    return mask_binary

# Streamlit app
st.title("Brain MRI Metastasis Segmentation")
st.write("Upload a brain MRI image to get the predicted segmentation mask.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    file_bytes = uploaded_file.read()
    image = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    if img is not None:
        st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        preprocessed_img = preprocess_image(img)

        # Perform prediction
        predicted_mask = model.predict(preprocessed_img)

        # Postprocess the predicted mask
        postprocessed_mask = postprocess_mask(predicted_mask[0])  # Get first sample

        # Display the predicted mask
        st.image(postprocessed_mask, caption="Predicted Mask", use_column_width=True, clamp=True)
    else:
        st.error("Error decoding the image.")

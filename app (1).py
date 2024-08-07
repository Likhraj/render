import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
 
# Load the trained model
model = tf.keras.models.load_model("kidney_model.keras")
 
# Define class labels
class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']
 
# Function to preprocess and predict image
def classify_image(image_path):
    # Preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    # Predict the class
    predictions = model.predict(image_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence
 
# Streamlit app
st.set_page_config(page_title="Kidney Image Classification", page_icon="🩺", layout="wide")
 
# Custom CSS for background color and navigation bar
st.markdown("""
<style>
/* Main content background */
.reportview-container, .main, .block-container {
background-image: url("/content/bg.jpg");
}
 
/* Sidebar background */
.sidebar .sidebar-content {
    background-color: red;
}
 
/* Align text to center */
.css-1aumxhk {
    text-align: center;
}
 
/* Navigation bar */
.navbar {
    overflow: hidden;
    background-color: DodgerBlue;
}
 
.navbar a {
    float: left;
    display: block;
    color: #f2f2f2;
    text-align: center;
    padding: 14px 16px;
    text-decoration: none;
}
 
.navbar a:hover {
    background-color: #ddd;
    color: black;
}
 
.navbar a.active {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)
 
# Navigation bar
st.markdown("""
<div class="navbar">
<a class="active" href="/">Home</a>
<a href="#about">About</a>
<a href="#contact">Contact</a>
</div>
""", unsafe_allow_html=True)
 
st.title("🩺 Kidney Image Classification")
st.write("Upload an image of a kidney, and the model will predict whether it is Normal, Cyst, Tumor, or Stone.")
 
# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
 
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
 
    # Save the uploaded image to a directory
    save_folder = 'uploaded_images'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    file_path = os.path.join(save_folder, uploaded_file.name)
 
    # Save the uploaded image
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
 
    # Classify the image
    predicted_class, confidence = classify_image(file_path)
 
    # Display the prediction
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
 
    # Add a progress bar to indicate the confidence level
    st.progress(int(confidence * 100))
 
# About section (can be expanded further)
st.markdown("""
<h2 id="about">About</h2>
<p>This application uses a deep learning model to classify kidney images into one of the following categories: Cyst, Normal, Stone, or Tumor.</p>
""", unsafe_allow_html=True)
 
# Contact section (can be expanded further)
st.markdown("""
<h2 id="contact">Contact</h2>
<p>If you have any questions or feedback, please contact us at example@example.com.</p>
""", unsafe_allow_html=True)
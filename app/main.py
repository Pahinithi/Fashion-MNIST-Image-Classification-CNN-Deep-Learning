import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load the pre-trained model
model_path = os.path.join(os.getcwd(), 'trained_model', 'trained_fashion_mnist_model.h5')
model = tf.keras.models.load_model(model_path)

# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 28, 28, 1)

# Custom CSS for a gradient background
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #d9a4a0, #bacfdb, #efd8f0);
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .stFileUploader > div {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 10px;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 14px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.title('Fashion Classifier')

st.text('Upload an image to classify it into one of the fashion categories.')

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(Image.open(uploaded_image).resize((150, 150)), caption='Uploaded Image')
    
    if st.button('Classify'):
        img_array = preprocess_image(uploaded_image)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        
        st.success(f'Prediction: {predicted_class}')

# Footer with developer credit
st.markdown(' <div class="footer"> <center> Developed by Nithilan</center></div>  ', unsafe_allow_html=True)

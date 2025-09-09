# Import necessary libraries
import streamlit as st
from fastai.vision.all import *
import gdown
import os
import platform
import pathlib
from pathlib import PosixPath

# URL of the model on Google Drive
model_url = 'https://drive.google.com/uc?id=1ELrjuiTUX5V3c1vFdqgD0NQkaCDU2p1_'
model_path = 'exportCNN10.pkl'

# Function to download and load the model, cached to avoid repeated downloads
@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)
        
    # Fix path issue when model is exported on Windows but run on Linux
    if platform.system() != "Windows":
        pathlib.WindowsPath = PosixPath 
    return load_learner(model_path, cpu=True)

# Load the model
learn = load_model()

# Create a Streamlit app
st.title('Waste Classification App')

# Add a file uploader
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'png', 'jpeg', 'jfif', 'webp'])

# Preprocess the uploaded image
if uploaded_file is not None:
    image = load_image(uploaded_file)

# Make predictions and display results
    st.write('Predicting...')
    pred, pred_idx, probs = learn.predict(image)
    disposal_type = {
        'battery': 'Special Handling',
        'biological': 'Compostable',
        'cardboard': 'Recyclable',
        'glass': 'Recyclable',
        'metal': 'Recyclable',
        'paper': 'Recyclable',
        'plastic': 'Recyclable/Landfill',
        'trash': 'Landfill'
    }
    st.write('Predicted Waste Class:', pred.capitalize())
    st.write('Confidence:', f'{probs[pred_idx].item() * 100:.2f}%')
    st.write('Disposal Instruction:', disposal_type[pred])

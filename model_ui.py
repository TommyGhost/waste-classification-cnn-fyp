# Import necessary libraries
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import numpy as np

# Load your FastAI model
learn = load_learner('exportCNN10.pkl')

# Create a Streamlit app
st.title('Waste Classification App')

# Add a file uploader
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'png', 'jpeg', 'jfif', 'webp'])

# Preprocess the uploaded image
if uploaded_file is not None:
    image = load_image(uploaded_file)

# Make predictions and display results
if st.button('Predict'):
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

import os
import gdown

MODEL_PATH = "cat_dog_classifier.h5"
FILE_ID = "1rpV6VOh_pXkfQgk2ih6hZqksYKe8Bt71?"  # replace this with your actual file ID
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Auto-download if model not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import os

# Load model with cache
@st.cache_resource
def load_model_cached():
    return load_model("cat_dog_classifier.h5")

model = load_model_cached()

# --- Sidebar Theme Toggle with Icons ---
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Theme", ("🌞 Light", "🌙 Dark"))

# Set background image based on theme
if "Light" in theme:
    bg_path = "bg_light.jpg"
else:
    bg_path = "bg_dark.jpg"  # dark theme

# Encode image for CSS background
def set_bg_from_local(path):
    with open(path, "rb") as f:
        img_data = f.read()
    encoded = base64.b64encode(img_data).decode()
    css = f"""
    <style>
    /* Target the main app container for the background */
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Target the main content area to remove default padding/margin */
    [data-testid="stAppViewContainer"] > .main {{
        background-color: rgba(0,0,0,0) !important; /* Make main content transparent */
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* Target the sidebar to make its background transparent and apply the image if desired */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(0,0,0,0) !important; /* Make sidebar transparent */
        /* If you want the background image to extend into the sidebar, uncomment the lines below */
        /*
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        */
    }}

    /* Remove padding from block container to prevent content from being pushed in */
    .block-container {{
        padding-left: 0rem;
        padding-right: 0rem;
        padding-top: 0rem;
        padding-bottom: 0rem;
    }}

    /* Remove the default header background if it's visible */
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0);
    }}

    /* Ensure the content is visible over the background */
    h1, h2, h3, h4, h5, h6, label, p, .stMarkdown, .stButton, .stFileUploader, .stRadio {{
        color: white; /* Adjust text color for readability on dark background */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8); /* Add text shadow for contrast */
    }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_from_local(bg_path)

# --- Title ---
st.title("🐶🐱 Cat vs Dog Classifier")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

# --- Sample image buttons ---
st.markdown("### Or try a sample:")
col1, col2 = st.columns(2)
with col1:
    if st.button("🐱 Sample Cat"):
        uploaded_file = "samples/cat.jpg"
with col2:
    if st.button("🐶 Sample Dog"):
        uploaded_file = "samples/dog.jpg"

# --- Prediction Function ---
def predict(img):
    img = img.resize((150, 150)).convert('RGB')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"

# --- Show Image, Predict, and Allow Download ---
if uploaded_file:
    if isinstance(uploaded_file, str):
        image = Image.open(uploaded_file)
    else:
        image = Image.open(uploaded_file)

    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    result = predict(image)
    st.markdown(f"### ✅ Prediction: **{result}**")

    # Download result
    result_text = f"Prediction: {result}"
    result_bytes = result_text.encode()
    b64 = base64.b64encode(result_bytes).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="prediction.txt">📄 Download Prediction</a>'
    st.markdown(href, unsafe_allow_html=True)
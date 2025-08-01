import os
import base64
import requests
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- Model Setup ---
MODEL_PATH = "cat_dog_classifier.h5"
HF_URL = "https://huggingface.co/ameyakj/CatsandDogclassifier/resolve/main/cat_dog_classifier.h5"

# Download model from Hugging Face if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model from Hugging Face..."):
        response = requests.get(HF_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

# --- Load model with cache ---
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

model = load_model_cached()

# [The rest of your code continues unchanged from here]


# --- Theme Toggle ---
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Theme", ("🌞 Light", "🌙 Dark"))
bg_path = "bg_light.jpg" if "Light" in theme else "bg_dark.jpg"

# --- Background Image Setup ---
def set_bg_from_local(path):
    with open(path, "rb") as f:
        img_data = f.read()
    encoded = base64.b64encode(img_data).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"] > .main {{
        background-color: rgba(0,0,0,0) !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(0,0,0,0) !important;
    }}
    .block-container {{
        padding: 0rem;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0);
    }}
    h1, h2, h3, h4, h5, h6, label, p, .stMarkdown, .stButton, .stFileUploader, .stRadio {{
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_from_local(bg_path)

# --- App Title ---
st.title("🐶🐱 Cat vs Dog Classifier")

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

# --- Ensure Sample Images Exist ---
os.makedirs("samples", exist_ok=True)
sample_cat_path = "samples/cat.jpg"
sample_dog_path = "samples/dog.jpg"

# Dummy placeholder images if not present
if not os.path.exists(sample_cat_path):
    Image.new('RGB', (150, 150), (255, 200, 200)).save(sample_cat_path)
if not os.path.exists(sample_dog_path):
    Image.new('RGB', (150, 150), (200, 200, 255)).save(sample_dog_path)

# --- Sample Image Buttons ---
st.markdown("### Or try a sample:")
col1, col2 = st.columns(2)
with col1:
    if st.button("🐱 Sample Cat"):
        uploaded_file = sample_cat_path
with col2:
    if st.button("🐶 Sample Dog"):
        uploaded_file = sample_dog_path

# --- Prediction Function ---
def predict(img):
    img = img.resize((150, 150)).convert('RGB')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array, verbose=0)
    return "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"

# --- Run Prediction and Show Output ---
if uploaded_file:
    image = Image.open(uploaded_file) if isinstance(uploaded_file, str) else Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    result = predict(image)
    st.markdown(f"### ✅ Prediction: **{result}**")

    # Download Result Button
    result_text = f"Prediction: {result}"
    b64 = base64.b64encode(result_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="prediction.txt">📄 Download Prediction</a>'
    st.markdown(href, unsafe_allow_html=True)

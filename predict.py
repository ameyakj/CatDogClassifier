import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = tf.keras.models.load_model("cat_dog_model.h5")

# Image path (you can change this to test other images)
img_path = "test_image.jpg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(150, 150))  # Use (224, 224) if using MobileNet/VGG etc.
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Predict
prediction = model.predict(img_array)
class_name = "Dog" if prediction[0][0] > 0.5 else "Cat"
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

print(f"Prediction: {class_name} ({confidence*100:.2f}%)")

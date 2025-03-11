import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

# Load the trained model
MODEL_PATH = "model/mobilenetv2_skin.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Load home remedies data
HOME_REMEDIES_PATH = "utils/home_remedies.json"
if not os.path.exists(HOME_REMEDIES_PATH):
    st.error(f"Home remedies file not found at {HOME_REMEDIES_PATH}. Please check the path.")
else:
    try:
        with open(HOME_REMEDIES_PATH, "r") as f:
            HOME_REMEDIES = json.load(f)
        st.success("Home remedies data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading home remedies: {e}")

# Class labels
CLASS_LABELS = [
    "Actinic keratosis", "Atopic Dermatitis", "Benign keratosis",
    "Candidiasis", "Dermatofibroma", "Melanoma", "Melanocytic nevus",
    "Squamous cell carcinoma", "Tinea", "Ringworm", "Vascular lesion"
]

# Streamlit app
st.title("Comprehensive Skin Disease Prediction System")
st.sidebar.header("Upload Image")

# Image upload
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image",  use_container_width=True)

        # Preprocess and predict
        try:
            from utils.preprocess import preprocess_image
            img_array = preprocess_image(uploaded_file)
            predictions = model.predict(img_array)
            predicted_class = CLASS_LABELS[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            # Display prediction
            st.subheader("Prediction")
            st.write(f"**Predicted Condition:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            # Display analysis
            st.subheader("Analysis")
            st.write(HOME_REMEDIES.get(predicted_class, {}).get("description", "No description available."))
            st.write(f"**Risk Level:** {HOME_REMEDIES.get(predicted_class, {}).get('caution', 'N/A')}")

            # Display home remedies
            st.subheader("Home Remedies")
            for remedy in HOME_REMEDIES.get(predicted_class, {}).get("remedies", []):
                st.write(f"- {remedy}")

            # Nearby dermatologists
            st.subheader("Nearby Dermatologists")
            try:
                from utils.geolocation import find_nearby_dermatologists, get_user_location
                user_location = get_user_location()  # Replace with user's location
                dermatologists = find_nearby_dermatologists(user_location)
                if dermatologists:
                    for doc in dermatologists:
                        st.write(f"**Name:** {doc['name']}")
                        st.write(f"**Address:** {doc['address']}")
                        st.write(f"**Distance:** {doc['distance']} km")
                        st.write("---")
                else:
                    st.write("No dermatologists found within 10 km.")
            except Exception as e:
                st.error(f"Error finding nearby dermatologists: {e}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    except Exception as e:
        st.error(f"Error loading image: {e}")
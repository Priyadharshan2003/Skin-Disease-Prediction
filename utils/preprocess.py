import tensorflow as tf
from tensorflow import keras
from keras.utils import load_img, img_to_array
import numpy as np

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocess the image for MobileNetV2 model.
    """
    # Build metrics before prediction to avoid warning
    if not hasattr(preprocess_image, 'model_initialized'):
        tf.keras.backend.clear_session()
        preprocess_image.model_initialized = True
        
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array
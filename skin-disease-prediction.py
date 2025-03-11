import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 11
TRAIN_DIR = 'Split_smol/train/'
VAL_DIR = 'Split_smol/val/'

# Class names
class_names = [
    'Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 
    'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 
    'Squamous cell carcinoma', 'Tinea Ringworm', 'Candidiasis', 
    'Vascular lesion'
]

# Data Generators with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model building using EfficientNetB0
def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train the model
model = build_model()

# Callbacks
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

callbacks = [checkpoint, early_stop, reduce_lr]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Fine-tuning - unfreeze some layers of the base model
base_model = model.layers[0]
base_model.trainable = True

# Freeze the first 200 layers and unfreeze the rest for fine-tuning
for layer in base_model.layers[:200]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training with fine-tuning
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=10,
    callbacks=callbacks
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

# Function to predict a single image
def predict_skin_disease(image_path):
    from tensorflow.keras.preprocessing import image
    
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    class_name = list(train_generator.class_indices.keys())[predicted_class]
    
    return {
        'class_name': class_name,
        'confidence': confidence,
        'all_probabilities': prediction[0]
    }

# Function to get home remedies for each skin condition (with caution)
def get_home_remedies(condition):
    remedies = {
        'Actinic keratosis': [
            "CAUTION: Actinic keratosis can develop into skin cancer. Medical attention is strongly advised.",
            "Use broad-spectrum sunscreen (SPF 30+) daily",
            "Wear protective clothing and wide-brimmed hats",
            "Apply aloe vera gel to soothe the affected area",
            "DO NOT attempt to remove or treat lesions yourself"
        ],
        'Atopic Dermatitis': [
            "Use gentle, fragrance-free moisturizers regularly",
            "Take lukewarm (not hot) showers",
            "Apply colloidal oatmeal baths to relieve itching",
            "Wear soft, cotton clothing",
            "Identify and avoid personal trigger factors"
        ],
        'Benign keratosis': [
            "CAUTION: Consult a dermatologist for proper diagnosis",
            "Gentle cleansing with mild soap",
            "Apply petroleum jelly to keep the area moisturized",
            "Use broad-spectrum sunscreen daily",
            "Do not attempt to remove lesions at home"
        ],
        'Dermatofibroma': [
            "CAUTION: Medical evaluation is recommended for proper diagnosis",
            "No home treatment is typically needed for dermatofibromas",
            "Use broad-spectrum sunscreen on the affected area",
            "Avoid trauma to the lesion",
            "Monitor for any changes in size or appearance"
        ],
        'Melanocytic nevus': [
            "CAUTION: Regular monitoring by a dermatologist is recommended",
            "Use broad-spectrum sunscreen daily",
            "Perform monthly self-examinations to monitor changes",
            "Avoid sun exposure during peak hours (10am-4pm)",
            "Document any changes in size, color, or shape"
        ],
        'Melanoma': [
            "URGENT: Melanoma is a serious form of skin cancer. Immediate medical attention is required.",
            "DO NOT attempt home remedies",
            "Seek immediate professional medical care",
            "Regular skin checks are essential for early detection",
            "Follow your doctor's treatment plan strictly"
        ],
        'Squamous cell carcinoma': [
            "URGENT: This is a form of skin cancer. Immediate medical attention is required.",
            "DO NOT attempt home remedies",
            "Seek immediate professional medical care",
            "Use sun protection consistently after treatment",
            "Follow your doctor's treatment plan strictly"
        ],
        'Tinea Ringworm': [
            "Over-the-counter antifungal creams (containing clotrimazole or miconazole)",
            "Keep the affected area clean and dry",
            "Avoid sharing personal items like towels or clothing",
            "Wash bedding and clothing in hot water",
            "Continue treatment for 1-2 weeks after symptoms resolve"
        ],
        'Candidiasis': [
            "Keep affected areas clean and dry",
            "Over-the-counter antifungal creams (containing clotrimazole)",
            "Wear loose-fitting cotton clothing",
            "Avoid scented products in the affected area",
            "Maintain good hygiene practices"
        ],
        'Vascular lesion': [
            "CAUTION: Medical evaluation is recommended for proper diagnosis",
            "Avoid trauma to the affected area",
            "Use sun protection",
            "Apply cool compresses if the area is uncomfortable",
            "Avoid blood thinners unless prescribed by your doctor"
        ]
    }
    
    return remedies.get(condition, ["No specific home remedies available. Please consult a dermatologist."])

# Function to find nearby dermatologists based on user location
def find_nearby_dermatologists(user_location, radius=10):
    try:
        # Initialize geolocator
        geolocator = Nominatim(user_agent="skin_disease_app")
        
        # Get coordinates for user location
        location = geolocator.geocode(user_location)
        if not location:
            return "Could not find your location. Please try another address or city name."
        
        user_coords = (location.latitude, location.longitude)
        
        # This would normally use an API like Google Places to find dermatologists
        # For demonstration, we'll use a mock database of dermatologists
        mock_dermatologists = [
            {"name": "Dr. Smith Dermatology Clinic", "address": "123 Health St", "city": "New York", "coords": (user_coords[0] + 0.02, user_coords[1] + 0.01), "rating": 4.8},
            {"name": "Skin Care Center", "address": "456 Wellness Ave", "city": "New York", "coords": (user_coords[0] - 0.01, user_coords[1] + 0.03), "rating": 4.5},
            {"name": "Advanced Dermatology", "address": "789 Medical Blvd", "city": "New York", "coords": (user_coords[0] + 0.04, user_coords[1] - 0.02), "rating": 4.7},
            {"name": "City Skin Specialists", "address": "101 Doctor's Lane", "city": "New York", "coords": (user_coords[0] - 0.03, user_coords[1] - 0.04), "rating": 4.2},
            {"name": "University Dermatology Clinic", "address": "202 Research Park", "city": "New York", "coords": (user_coords[0] + 0.05, user_coords[1] + 0.05), "rating": 4.9}
        ]
        
        # Calculate distances and filter by radius
        nearby_dermatologists = []
        for derm in mock_dermatologists:
            distance = geodesic(user_coords, derm["coords"]).kilometers
            if distance <= radius:
                derm["distance"] = round(distance, 2)
                nearby_dermatologists.append(derm)
        
        # Sort by distance
        nearby_dermatologists.sort(key=lambda x: x["distance"])
        
        return nearby_dermatologists
    
    except Exception as e:
        return f"Error finding nearby dermatologists: {str(e)}"

# Function to create a map with nearby dermatologists
def create_dermatologist_map(user_location, dermatologists):
    try:
        geolocator = Nominatim(user_agent="skin_disease_app")
        location = geolocator.geocode(user_location)
        
        if not location:
            return None
            
        user_coords = (location.latitude, location.longitude)
        
        # Create map centered at user location
        m = folium.Map(location=user_coords, zoom_start=13)
        
        # Add marker for user location
        folium.Marker(
            user_coords,
            popup="Your Location",
            icon=folium.Icon(color="blue", icon="home")
        ).add_to(m)
        
        # Add markers for dermatologists
        for derm in dermatologists:
            folium.Marker(
                derm["coords"],
                popup=f"{derm['name']}<br>{derm['address']}<br>Rating: {derm['rating']}⭐<br>Distance: {derm['distance']} km",
                icon=folium.Icon(color="red", icon="plus")
            ).add_to(m)
            
        # Draw circles for the 10km radius
        folium.Circle(
            user_coords,
            radius=10000,  # 10km in meters
            color="green",
            fill=True,
            fill_opacity=0.1
        ).add_to(m)
        
        # Save the map
        map_file = "nearby_dermatologists.html"
        m.save(map_file)
        
        return map_file
    
    except Exception as e:
        return None

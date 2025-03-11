import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np

# Define paths to your dataset
TRAIN_DATA_DIR = "data/Split_smol/train"
VAL_DATA_DIR = "data/Split_smol/val"

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for validation, just rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    VAL_DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Load MobileNetV2 as the base model
base_model = MobileNetV2(
    include_top=False,             # Exclude the top classification layer
    weights="imagenet",            # Use pre-trained weights from ImageNet
    input_shape=(224, 224, 3)      # Input image shape
)
base_model.trainable = False       # Freeze the base model

# Add custom layers on top of the base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),      # Global average pooling
    Dense(128, activation="relu"), # Fully connected layer
    Dropout(0.5),                 # Dropout for regularization
    Dense(9, activation="softmax") # Output layer for 9 classes
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Print model summary
model.summary()

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[lr_scheduler]
)

# Save the trained model
model.save("model/mobilenetv2_skin.h5")
print("Model saved to model/mobilenetv2_skin.h5")
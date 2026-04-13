# Run this line to install dependencies if you are in Google Colab:
# !pip install tf2onnx onnx tensorflow-datasets

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import json

# --- PROJECT INFORMATION ---
# Project: PlantDoc Universal Disease Diagnosis
# Dataset: PlantVillage (Open-access repository of 54,300+ images)
# Model: MobileNetV2 Transfer Learning
# Target: 38 Disease Classes 🌿

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_NAME = "plant_disease_model"

def build_model(num_classes):
    """ Builds a MobileNetV2-based model for transfer learning """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Freeze base for initial training

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    print("--- PlantDoc Universal Trainer: PlantVillage ---")
    
    # Load the official PlantVillage dataset via TFDS
    print("Fetching dataset... (This may take a few minutes)")
    (ds_train, ds_val), ds_info = tfds.load(
        'plant_village',
        split=['train[:80%]', 'train[80%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    num_classes = ds_info.features['label'].num_classes
    class_names = ds_info.features['label'].names
    print(f"Dataset Loaded: {num_classes} classes found.")

    def preprocess(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, num_classes)
        return image, label

    # Data Augmentation & Preprocessing
    train_ds = ds_train.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = ds_val.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build and Train
    model = build_model(num_classes)
    print("Starting training... (Targeting 38 disease classes)")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Save Class Names for the Backend
    class_names_dict = {i: name for i, name in enumerate(class_names)}
    with open("class_names.json", "w") as f:
        json.dump(class_names_dict, f, indent=4)
    print("Success: class_names.json generated.")


    # Save as Keras
    model.save(f"{MODEL_NAME}.keras")
    print(f"Success: {MODEL_NAME}.keras saved.")

    # Conversion to ONNX (Requires tf2onnx)
    # pip install tf2onnx
    import tf2onnx
    import onnx

    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    output_path = f"{MODEL_NAME}.onnx"
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(model_proto, output_path)
    print(f"Success: {MODEL_NAME}.onnx exported. You can now use this in the PlantDoc backend!")

if __name__ == "__main__":
    train()

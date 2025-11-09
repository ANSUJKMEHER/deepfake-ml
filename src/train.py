import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from src.feature_extraction import get_feature_extractor
from src.preprocess import load_data
from src.model import build_model


def main():
    # Paths
    train_dir = "data/train_images"
    json_path = "data/train.json"
    model_save_path = "models/deepfake_predictor.h5"
    os.makedirs("models", exist_ok=True)

    print("📦 Loading and preprocessing data...")
    images, labels, _ = load_data(train_dir, json_path)
    print(f"✅ Loaded {len(images)} images successfully.")
    print("📊 Label distribution:", Counter(labels))

    # Feature extractor (EfficientNetB4 fine-tuned)
    extractor = get_feature_extractor(trainable_layers=100)
    preprocess_fn = getattr(extractor, "preprocess_input", None)
    if preprocess_fn is None:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        preprocess_fn = preprocess_input

    # Data augmentation
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    print("🔍 Extracting image features...")
    images = datagen.standardize(images)
    features = extractor.predict(images, batch_size=32, verbose=1)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print("🧠 Building model...")
    model = build_model(features.shape[1])

    # Callbacks
    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

    # Training
    print("🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )

    print("✅ Training complete! Model saved at:", model_save_path)

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/training_curve.png")

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/loss_curve.png")


if __name__ == "__main__":
    main()

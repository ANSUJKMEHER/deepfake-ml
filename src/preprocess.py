import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def load_data(image_dir, json_path=None, img_size=(380, 380)):  # EfficientNetB4 expects 380x380
    images = []
    labels = []
    image_names = []

    # Load JSON data
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    print(f"🧾 Found {len(data)} entries in {json_path}" if json_path else f"Scanning {image_dir} for images")

    for entry in tqdm(data, desc="Loading images"):
        img_path = os.path.join(image_dir, entry["image_path"])
        if not os.path.exists(img_path):
            print(f"⚠️ Missing image: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Warning: could not read image {img_path}")
            continue

        # Resize for EfficientNetB4 input
        img = cv2.resize(img, img_size)

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize pixel values (EfficientNet expects floats)
        # img = img.astype('float32') / 255.0

        images.append(img)
        labels.append(entry["label"])
        image_names.append(os.path.basename(img_path))

    images = np.array(images)
    labels = np.array(labels)

    print(f"✅ Loaded {len(images)} images successfully.")
    return images, labels, image_names

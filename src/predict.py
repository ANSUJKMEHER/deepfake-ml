# src/predict.py  (overwrite with this)
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



import os
import json
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from tqdm import tqdm
from src.feature_extraction import get_feature_extractor

def load_test_images_flat(test_dir, target_size=(380, 380)):
    """Load all images from a flat test folder (no subdirectories).
       Return raw image arrays (dtype float32) — preprocessing applied later.
    """
    imgs = []
    names = []

    for fname in tqdm(sorted(os.listdir(test_dir)), desc="Loading test images"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(test_dir, fname)
            try:
                img = image.load_img(path, target_size=target_size)
                img_array = image.img_to_array(img).astype('float32')
                imgs.append(img_array)
                names.append(fname)
            except Exception as e:
                print(f"⚠️ Skipping {fname}: {e}")

    return np.array(imgs), names

def main():
    team_name = "ansujkmeher"
    test_dir = "data/test"
    model_path = "models/deepfake_predictor.h5"
    output_path = f"outputs/{team_name}_prediction.json"

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"❌ Test directory not found: {test_dir}")

    print("🧠 Loading test images...")
    images, image_names = load_test_images_flat(test_dir, target_size=(380, 380))

    if len(images) == 0:
        raise ValueError(f"❌ No images found in {test_dir}. Please ensure it contains .png/.jpg files.")

    print(f"✅ Loaded {len(images)} images successfully.")
    print("🔍 Extracting features from test images...")

    # get extractor and the exact preprocess function used during training
    extractor = get_feature_extractor()
    preprocess_fn = getattr(extractor, "preprocess_input", None)
    if preprocess_fn is None:
        # fallback
        from tensorflow.keras.applications.efficientnet import preprocess_input
        preprocess_fn = preprocess_input

    # apply identical preprocessing used in training
    images_pre = np.array([preprocess_fn(x) for x in images], dtype=np.float32)

    features = extractor.predict(images_pre, batch_size=32, verbose=1)

    print("📦 Loading trained model...")
    model = load_model(model_path)

    print("🧮 Making predictions...")
    preds = model.predict(features, verbose=1).flatten()

    # Save results as mapping filename -> predicted probability (0..1)
    results = {name: float(p) for name, p in zip(image_names, preds)}

    os.makedirs("outputs", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"✅ Predictions saved to: {output_path}")
    print("\n📊 Sample Predictions:")
    for name, prob in list(results.items())[:10]:
        label = "FAKE" if prob < 0.5 else "REAL"
        print(f"{name} → {label} ({prob:.4f})")

    # helpful stats
    print("\n🔎 Prediction stats -> min, max, mean:", float(preds.min()), float(preds.max()), float(preds.mean()))

if __name__ == "__main__":
    main()

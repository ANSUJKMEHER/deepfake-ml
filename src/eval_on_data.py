# src/eval_on_data.py
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from src.preprocess import load_data
from src.feature_extraction import get_feature_extractor
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

imgs, labels, names = load_data("data/train_images", "data/train.json")
print("Loaded labeled images:", len(imgs), "Label dist:", Counter(labels))

# preprocess exactly like inference
imgs_pre = np.array([preprocess_input(x) for x in imgs], dtype=np.float32)
ext = get_feature_extractor()
feats = ext.predict(imgs_pre, batch_size=32, verbose=1)

m = load_model("models/deepfake_predictor.h5")
probs = m.predict(feats, verbose=1).flatten()
y_true = np.array(labels).astype(int)
y_pred = (probs >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_true, y_pred))
try:
    print("ROC AUC:", roc_auc_score(y_true, probs))
except Exception as e:
    print("ROC AUC error:", e)
print("Confusion matrix (TN FP / FN TP):")
print(confusion_matrix(y_true, y_pred))

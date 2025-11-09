# src/check_data_load.py
import sys, os
# add project root to sys.path so "src" package can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import load_data
from collections import Counter

images, labels, names = load_data("data/train_images", "data/train.json")
print("Loaded images:", len(images))
print("Label counts (loaded):", Counter(labels))
print("Example file names:", names[:5])

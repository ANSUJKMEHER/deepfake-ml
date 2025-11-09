# Deepfake ML Model Hackathon (Synergy’25)

## Overview
This project predicts deepfake detection outputs for unseen images using deep learning.
It uses EfficientNetB0 for feature extraction and a regression model to predict similarity
to proprietary deepfake outputs.

## Steps to Run
1. Place training data in `data/train_images/` and `data/train.json`.
2. Place test images in `data/test_images/`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

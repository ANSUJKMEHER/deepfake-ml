
```markdown
# 🧠 Deepfake Detection Model – Synergy’25 Hackathon

## 📌 Overview
This project implements a **Deepfake Detection System** designed for the **Synergy’25 Hackathon**.  
It uses **EfficientNetB4** as a feature extractor and a custom **dense neural network** classifier to distinguish between **real** and **fake (AI-generated)** images with high accuracy.

The model achieves **~97% accuracy** and **0.9949 ROC-AUC**, demonstrating strong performance and robustness on unseen test data.

---

## ⚙️ Tech Stack
- **Framework:** TensorFlow / Keras  
- **Language:** Python 3.10  
- **Backbone:** EfficientNetB4 (Transfer Learning)  
- **Optimizer:** Adam (lr = 3e-5)  
- **Loss Function:** Binary Crossentropy  
- **Regularization:** Dropout + L2 Regularization  

---

## 📁 Folder Structure

```

📁 deepfake-ml/
  ├─ models/
  │   ├─ deepfake_predictor.h5             # Trained model weights
  │   ├─ evaluation_summary.txt            # Metrics summary
  │   ├─ training_curve.png
  │   ├─ loss_curve.png
  │   ├─ roc_curve.png
  │   └─ confusion_matrix.png
  │
  ├─ outputs/
  │   └─ ansujkmeher_prediction.json       # Final test predictions
  │
  ├─ src/
  │   ├─ model.py                          # Classifier architecture
  │   ├─ feature_extraction.py             # EfficientNetB4 extractor
  │   ├─ preprocess.py                     # Image preprocessing
  │   ├─ train.py                          # Training pipeline
  │   ├─ predict.py                        # Inference script
  │   ├─ eval_on_data.py                   # Evaluation script
  │   ├─ evaluate.py                       # Alternate evaluation script
  │   ├─ prepare_train_json.py             # Combine fake/real JSON to train.json
  │   └─ check_*                           # Sanity check scripts
  │
  ├─ requirements.txt                      # Python dependencies
  ├─ app.ipynb                             # Optional notebook for analysis
  └─ README.md                             # Project documentation

````

---

## 🚀 Steps to Run

### 1️⃣ Setup Environment
```bash
git clone https://github.com/<your-username>/deepfake-ml.git
cd deepfake-ml
python -m venv .venv
.venv\Scripts\activate        # Windows
# or
source .venv/bin/activate     # Linux / Mac
pip install -r requirements.txt
````

---

### 2️⃣ Prepare Dataset

Ensure this structure:

```
data/
│
├── train_images/
│   ├── fake_cifake_images/
│   └── real_cifake_images/
│
├── fake_cifake_preds.json
├── real_cifake_preds.json
└── test/
    ├── 1.png
    ├── 2.png
    ├── 3.png
    └── ...
```

Then generate the combined training metadata:

```bash
python src/prepare_train_json.py
```

---

### 3️⃣ Train the Model

```bash
python src/train.py
```

This will:

* Load and augment data
* Extract features via EfficientNetB4
* Train the dense classifier
* Save best weights as `models/deepfake_predictor.h5`

---

### 4️⃣ Evaluate Model

```bash
python src/eval_on_data.py
```

Generates:

* Accuracy, ROC-AUC, and confusion matrix
* Plots in `/models` and summary in `evaluation_summary.txt`

---

### 5️⃣ Generate Predictions

```bash
python src/predict.py
```

Outputs predictions for unseen test images:

```
outputs/ansujkmeher_prediction.json
```

Example JSON:

```json
{
  "1.png": 0.2291,
  "2.png": 0.8457,
  "3.png": 0.1052
}
```

👉 **Values near 1 → REAL**, **near 0 → FAKE**.

---

## 📊 Results Summary

| Metric           | Score                  |
| ---------------- | ---------------------- |
| Accuracy         | **96.95%**             |
| ROC-AUC          | **0.9949**             |
| Confusion Matrix | [[977, 23], [38, 962]] |

---

## 🧩 Methodology Overview

1. **Data Preparation:** Combine real/fake sets into one JSON mapping (`train.json`).
2. **Preprocessing:** Resize → Normalize → Augment images.
3. **Feature Extraction:** EfficientNetB4 pretrained on ImageNet.
4. **Classifier:** 3-layer dense network with dropout and L2 regularization.
5. **Optimization:** Binary crossentropy + Adam optimizer.
6. **Validation:** EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.
7. **Prediction:** Generate test results as JSON with probability scores.

---

## 💾 Deliverables

| File                                  | Description                  |
| ------------------------------------- | ---------------------------- |
| `models/deepfake_predictor.h5`        | Trained TensorFlow model     |
| `outputs/ansujkmeher_prediction.json` | Final predictions            |
| `models/*.png`                        | Training visualization plots |
| `models/evaluation_summary.txt`       | Evaluation results summary   |

---

## 👨‍💻 Author

**Ansuj K. Meher**
Deepfake ML Model Hackathon – Synergy’25
Developed using TensorFlow, Keras & Python

---

## 🏁 License

**MIT License** – for educational and research use.

---


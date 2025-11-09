import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from tensorflow.keras.models import load_model
from src.feature_extraction import get_feature_extractor
from src.preprocess import load_data

def evaluate_model():
    model_path = "models/deepfake_predictor.h5"
    data_dir = "data/train_images"
    json_path = "data/train.json"

    print("📦 Loading images...")
    images, labels, image_names = load_data(data_dir, json_path)
    print(f"✅ Loaded {len(images)} images successfully.")

    # Feature extraction (use same preprocessing as in training)
    print("🔍 Extracting features...")
    extractor = get_feature_extractor(trainable_layers=100)
    preprocess_fn = extractor.preprocess_input
    images = preprocess_fn(images)
    features = extractor.predict(images, batch_size=32, verbose=1)

    # Load trained model
    print("🧠 Loading trained model...")
    model = load_model(model_path)

    # Predictions
    print("🔮 Predicting...")
    preds = model.predict(features, verbose=1)
    y_pred = (preds > 0.5).astype("int32").flatten()
    y_true = np.array(labels).flatten()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    # Classification report
    report = classification_report(y_true, y_pred, target_names=["Fake", "Real"])
    print("\n📈 Classification Report:\n", report)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("models/roc_curve.png")
    plt.close()

    # Save summary
    with open("models/evaluation_summary.txt", "w") as f:
        f.write("Deepfake Detection Model Evaluation\n")
        f.write("=" * 40 + "\n\n")
        f.write(report + "\n")
        f.write(f"AUC Score: {roc_auc:.4f}\n")

    print(f"\n✅ Evaluation complete! AUC: {roc_auc:.4f}")
    print("📊 Results saved to:")
    print(" - models/confusion_matrix.png")
    print(" - models/roc_curve.png")
    print(" - models/evaluation_summary.txt")

if __name__ == "__main__":
    evaluate_model()

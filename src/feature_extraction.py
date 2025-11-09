from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model

def get_feature_extractor(trainable_layers=50):
    base_model = EfficientNetB4(weights='imagenet', include_top=False, pooling='avg')

    # Freeze most layers except the last N
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True

    print(f"🔧 Fine-tuning last {trainable_layers} layers of EfficientNetB4")

    model = Model(inputs=base_model.input, outputs=base_model.output)
    model.preprocess_input = preprocess_input  # Correct preprocessing function

    return model

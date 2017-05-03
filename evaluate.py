"""Evaluate model and plot confusion matrix."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


def evaluate(model_path, data_path):
    model = load_model(model_path)

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = datagen.flow_from_directory(
        data_path, target_size=(224, 224), batch_size=32, shuffle=False
    )

    preds = model.predict_generator(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.h5")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    evaluate(args.model, args.data)

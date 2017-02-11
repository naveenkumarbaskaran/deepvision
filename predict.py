"""Run inference on a single image."""
import argparse
import json
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


def predict(model_path, img_path, class_map_path="classes.json"):
    model = load_model(model_path)

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    top_idx = np.argsort(preds)[::-1][:5]

    with open(class_map_path) as f:
        classes = json.load(f)

    print("Top predictions:")
    for idx in top_idx:
        print(f"  {classes[str(idx)]}: {preds[idx]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="model.h5")
    args = parser.parse_args()
    predict(args.model, args.image)

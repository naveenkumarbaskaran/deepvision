"""Train a CNN image classifier using transfer learning on VGG16."""
import argparse
import os
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


def build_model(num_classes):
    base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze convolutional layers
    for layer in base.layers:
        layer.trainable = False

    x = Flatten()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base.input, outputs=predictions)


def main():
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="model.h5")
    args = parser.parse_args()

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        args.data, target_size=(224, 224), batch_size=args.batch_size, subset="training"
    )
    val_gen = datagen.flow_from_directory(
        args.data, target_size=(224, 224), batch_size=args.batch_size, subset="validation"
    )

    num_classes = train_gen.num_classes
    model = build_model(num_classes)
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])

    print(f"Training on {train_gen.samples} images, {num_classes} classes")
    model.fit_generator(train_gen, epochs=args.epochs, validation_data=val_gen)
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()

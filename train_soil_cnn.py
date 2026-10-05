import json
from pathlib import Path

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
train_dir = BASE_DIR / "dataset" / "Train"
test_dir = BASE_DIR / "dataset" / "test"
model_path = BASE_DIR / "soil_cnn_model.h5"
labels_path = BASE_DIR / "soil_labels.json"

img_height, img_width = 128, 128
batch_size = 16

if not train_dir.exists() or not test_dir.exists():
    raise FileNotFoundError(
        "Expected soil image folders at dataset/Train and dataset/test."
    )

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

num_classes = len(train_gen.class_indices)
model = models.Sequential(
    [
        layers.Input(shape=(img_height, img_width, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_gen,
    epochs=15,
    validation_data=test_gen,
)

model.save(model_path)

soil_labels = [None] * num_classes
for label, index in train_gen.class_indices.items():
    soil_labels[index] = label
labels_path.write_text(json.dumps(soil_labels, indent=2), encoding="utf-8")

loss, accuracy = model.evaluate(test_gen, verbose=0)
print(f"Model saved to {model_path}")
print(f"Soil labels saved to {labels_path}")
print(f"Test accuracy: {accuracy:.2f}")

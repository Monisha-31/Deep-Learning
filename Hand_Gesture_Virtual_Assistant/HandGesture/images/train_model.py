import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# 1. DATASET PATH
# ============================================================

DATASET_PATH = os.path.expanduser(
    r"~\Downloads\Hand Gesture\Images"
)

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    print("Dataset folder not found:")
    print(DATASET_PATH)
    exit()

print("Dataset found at:")
print(DATASET_PATH)


# ============================================================
# 3. DATA AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,

    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)


# ============================================================
# 4. TRAINING DATA
# ============================================================

train_data = datagen.flow_from_directory(
    DATASET_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True
)


# ============================================================
# 5. VALIDATION DATA
# ============================================================

validation_data = datagen.flow_from_directory(
    DATASET_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False
)


# ============================================================
# 6. DISPLAY CLASS NAMES
# ============================================================

class_names = list(train_data.class_indices.keys())

print("\nClasses detected:")

for index, name in enumerate(class_names):
    print(index, "->", name)


# ============================================================
# 7. CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Fourth convolution block
    layers.Conv2D(
        256,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Flatten
    layers.Flatten(),

    # Dense layers
    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(0.5),

    # 10 gesture classes
    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 9. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    train_data,

    validation_data=validation_data,

    epochs=EPOCHS
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

model.save("hand_gesture_model.keras")

print("\nModel saved as:")
print("hand_gesture_model.keras")


# ============================================================
# 12. SAVE CLASS NAMES
# ============================================================

with open(
    "gesture_classes.json",
    "w"
) as file:

    json.dump(
        class_names,
        file
    )

print("Class names saved as:")
print("gesture_classes.json")


# ============================================================
# 13. FINAL ACCURACY
# ============================================================

final_train_accuracy = history.history["accuracy"][-1]

final_validation_accuracy = history.history["val_accuracy"][-1]

print("\n==============================")
print("Training completed")
print("==============================")

print(
    f"Training Accuracy: "
    f"{final_train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy: "
    f"{final_validation_accuracy * 100:.2f}%"
)
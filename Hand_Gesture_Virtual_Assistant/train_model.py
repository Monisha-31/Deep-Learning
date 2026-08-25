import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix


# =========================================================
# 1. DATASET PATH
# =========================================================

DATASET_PATH = r"C:\Users\MONISHA\Downloads\HandGesture\images"


# =========================================================
# 2. SETTINGS
# =========================================================

IMG_WIDTH = 128
IMG_HEIGHT = 128

BATCH_SIZE = 32

EPOCHS = 8


# =========================================================
# 3. CHECK DATASET PATH
# =========================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Dataset path not found!")

    print("\nThe program is looking for:")
    print(DATASET_PATH)

    print("\nPlease check the path.")

    exit()


print("\nDataset found successfully!")


# =========================================================
# 4. DATA AUGMENTATION
# =========================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0,

    validation_split=0.20,

    rotation_range=15,

    width_shift_range=0.15,

    height_shift_range=0.15,

    zoom_range=0.15,

    shear_range=0.10,

    horizontal_flip=True

)


# =========================================================
# 5. TRAINING DATA
# =========================================================

train_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True

)


# =========================================================
# 6. VALIDATION DATA
# =========================================================

validation_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False

)


# =========================================================
# 7. DISPLAY DATASET INFORMATION
# =========================================================

print("\n==============================================")
print("DATASET INFORMATION")
print("==============================================")

print(
    "Training images:",
    train_generator.samples
)

print(
    "Validation images:",
    validation_generator.samples
)

print(
    "Number of classes:",
    len(train_generator.class_indices)
)


print("\nClass Mapping:")

print(train_generator.class_indices)


# =========================================================
# 8. SAVE CLASS NAMES
# =========================================================

class_names = list(
    train_generator.class_indices.keys()
)


with open(
    "gesture_classes.json",
    "w"
) as file:

    json.dump(
        class_names,
        file
    )


print("\nGesture classes:")

for i, name in enumerate(class_names):

    print(
        i,
        "->",
        name
    )


# =========================================================
# 9. BUILD CNN MODEL
# =========================================================

model = Sequential()


# First Convolution Block

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        )
    )
)

model.add(
    BatchNormalization()
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# Second Convolution Block

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)

model.add(
    BatchNormalization()
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# Third Convolution Block

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    )
)

model.add(
    BatchNormalization()
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# Fourth Convolution Block

model.add(
    Conv2D(
        256,
        (3, 3),
        activation="relu"
    )
)

model.add(
    BatchNormalization()
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# =========================================================
# 10. CLASSIFICATION LAYERS
# =========================================================

model.add(
    Flatten()
)

model.add(
    Dense(
        256,
        activation="relu"
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        len(class_names),
        activation="softmax"
    )
)


# =========================================================
# 11. COMPILE MODEL
# =========================================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# =========================================================
# 12. DISPLAY MODEL
# =========================================================

print("\n==============================================")
print("CNN MODEL")
print("==============================================")

model.summary()


# =========================================================
# 13. CALLBACKS
# =========================================================

early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=3,

    restore_best_weights=True

)


reduce_learning_rate = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=0.000001

)


# =========================================================
# 14. TRAIN MODEL
# =========================================================

print("\n==============================================")
print("TRAINING STARTED")
print("==============================================")

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        reduce_learning_rate
    ]

)


# =========================================================
# 15. EVALUATE MODEL
# =========================================================

print("\n==============================================")
print("EVALUATING MODEL")
print("==============================================")


loss, accuracy = model.evaluate(
    validation_generator
)


print(
    "\nValidation Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print(
    "Validation Loss:",
    round(
        loss,
        4
    )
)


# =========================================================
# 16. CLASSIFICATION REPORT
# =========================================================

validation_generator.reset()


predictions = model.predict(
    validation_generator
)


predicted_classes = np.argmax(
    predictions,
    axis=1
)


true_classes = validation_generator.classes


print("\n==============================================")
print("CLASSIFICATION REPORT")
print("==============================================\n")


print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)


# =========================================================
# 17. CONFUSION MATRIX
# =========================================================

print("\n==============================================")
print("CONFUSION MATRIX")
print("==============================================\n")


print(
    confusion_matrix(
        true_classes,
        predicted_classes
    )
)


# =========================================================
# 18. SAVE MODEL
# =========================================================

model.save(
    "hand_gesture_model.keras"
)


print("\n==============================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("==============================================")


print(
    "\nModel saved as:"
)

print(
    "hand_gesture_model.keras"
)


print(
    "\nClass names saved as:"
)

print(
    "gesture_classes.json"
)


print(
    "\nYou DO NOT need to train the model again."
)


print(
    "\nTo start the camera, run:"
)

print(
    "python realtime_assistant.py"
)
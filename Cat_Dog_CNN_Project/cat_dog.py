import os
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =========================================================
# 1. DATASET PATH
# =========================================================

train_dir = r"C:/Users/MONISHA/Downloads/archive (1)/dataset/train"
test_dir = r"C:/Users/MONISHA/Downloads/archive (1)/dataset/test"


# =========================================================
# 2. IMAGE SETTINGS
# =========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


# =========================================================
# 3. DATA AUGMENTATION
# =========================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.20,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# =========================================================
# 4. LOAD TRAINING DATA
# =========================================================

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)


# =========================================================
# 5. LOAD TEST DATA
# =========================================================

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


print("\nClass Mapping:")
print(train_generator.class_indices)


# =========================================================
# 6. LOAD PRETRAINED MOBILENETV2
# =========================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze MobileNetV2 layers
base_model.trainable = False


# =========================================================
# 7. CREATE MODEL
# =========================================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.4),

    Dense(
        1,
        activation="sigmoid"
    )
])


# =========================================================
# 8. COMPILE MODEL
# =========================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 9. DISPLAY MODEL
# =========================================================

model.summary()


# =========================================================
# 10. CALLBACKS
# =========================================================

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=1,
    min_lr=0.000001
)


# =========================================================
# 11. TRAIN MODEL
# =========================================================

print("\n================================")
print("TRAINING STARTED")
print("================================\n")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=5,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)


# =========================================================
# 12. EVALUATE MODEL
# =========================================================

loss, accuracy = model.evaluate(test_generator)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# =========================================================
# 13. SAVE MODEL
# =========================================================

model.save("cat_dog_mobilenetv2.keras")

print("\nModel saved successfully!")


# =========================================================
# 14. REAL-TIME WEBCAM
# =========================================================

print("\n================================")
print("STARTING CAMERA")
print("================================")

print("Show a CAT or DOG.")
print("Press Q to exit.")


cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

else:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Cannot read camera.")

            break


        # =================================================
        # PREPROCESS CAMERA IMAGE
        # =================================================

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize
        resized = cv2.resize(
            rgb_frame,
            IMG_SIZE
        )

        # Normalize
        image = resized.astype(
            "float32"
        ) / 255.0

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )


        # =================================================
        # PREDICTION
        # =================================================

        prediction = model.predict(
            image,
            verbose=0
        )[0][0]


        # =================================================
        # CAT / DOG CLASSIFICATION
        # =================================================

        if prediction >= 0.5:

            label = "DOG"

            confidence = prediction

        else:

            label = "CAT"

            confidence = 1 - prediction


        confidence_percentage = confidence * 100


        # =================================================
        # DISPLAY RESULT
        # =================================================

        text = f"{label} - {confidence_percentage:.1f}%"


        cv2.rectangle(
            frame,
            (10, 10),
            (400, 80),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            text,
            (25, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )


        # Display camera
        cv2.imshow(
            "Real-Time Cat and Dog Classification",
            frame
        )


        # =================================================
        # PRESS Q TO EXIT
        # =================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# =========================================================
# 15. CLOSE CAMERA
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("\nCamera closed.")
print("Program completed successfully.")
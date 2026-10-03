import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# =========================
# SETTINGS
# =========================

IMG_SIZE = 128

GRAY_PATH = "dataset/gray"
COLOR_PATH = "dataset/color"

MODEL_PATH = "models/colorization_autoencoder.keras"

EPOCHS = 2          # First test: keep this at 2
BATCH_SIZE = 16


# =========================
# GET IMAGE FILES
# =========================

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

gray_files = {
    os.path.splitext(f)[0]: os.path.join(GRAY_PATH, f)
    for f in os.listdir(GRAY_PATH)
    if f.lower().endswith(valid_extensions)
}

color_files = {
    os.path.splitext(f)[0]: os.path.join(COLOR_PATH, f)
    for f in os.listdir(COLOR_PATH)
    if f.lower().endswith(valid_extensions)
}

# Only use images that exist in BOTH folders
common_names = sorted(
    set(gray_files.keys()) & set(color_files.keys()),
    key=lambda x: int(x) if x.isdigit() else x
)

print("Total paired images:", len(common_names))

if len(common_names) == 0:
    raise RuntimeError(
        "No matching image pairs found in dataset/gray and dataset/color."
    )


# =========================
# TRAIN / TEST SPLIT
# =========================

train_names, test_names = train_test_split(
    common_names,
    test_size=0.2,
    random_state=42
)

print("Training images:", len(train_names))
print("Testing images :", len(test_names))


# =========================
# DATA GENERATOR
# =========================

class ColorizationSequence(tf.keras.utils.Sequence):

    def __init__(self, names, gray_files, color_files,
                 batch_size=16, img_size=128, shuffle=True):
        
        self.names = names
        self.gray_files = gray_files
        self.color_files = color_files
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.names))

    def __len__(self):
        return int(np.ceil(len(self.names) / self.batch_size))

    def __getitem__(self, index):

        batch_indexes = self.indexes[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_names = [
            self.names[i] for i in batch_indexes
        ]

        X = []
        Y = []

        for name in batch_names:

            # Read grayscale image
            gray = cv2.imread(
                self.gray_files[name],
                cv2.IMREAD_GRAYSCALE
            )

            # Read color image
            color = cv2.imread(
                self.color_files[name],
                cv2.IMREAD_COLOR
            )

            if gray is None or color is None:
                continue

            # Resize
            gray = cv2.resize(
                gray,
                (self.img_size, self.img_size)
            )

            color = cv2.resize(
                color,
                (self.img_size, self.img_size)
            )

            # OpenCV BGR → RGB
            color = cv2.cvtColor(
                color,
                cv2.COLOR_BGR2RGB
            )

            # Normalize
            gray = gray.astype(np.float32) / 255.0
            color = color.astype(np.float32) / 255.0

            # Add channel dimension
            gray = np.expand_dims(gray, axis=-1)

            X.append(gray)
            Y.append(color)

        return np.array(X), np.array(Y)

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes)


# =========================
# CREATE DATA GENERATORS
# =========================

train_generator = ColorizationSequence(
    train_names,
    gray_files,
    color_files,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE
)

test_generator = ColorizationSequence(
    test_names,
    gray_files,
    color_files,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    shuffle=False
)


# =========================
# BUILD AUTOENCODER
# =========================

model = models.Sequential([

    # -------------------------
    # ENCODER
    # -------------------------

    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 1)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    # -------------------------
    # DECODER
    # -------------------------

    layers.UpSampling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.UpSampling2D((2, 2)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    # RGB OUTPUT
    layers.Conv2D(
        3,
        (3, 3),
        activation="sigmoid",
        padding="same"
    )
])


# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# =========================
# SHOW MODEL
# =========================

model.summary()


# =========================
# CREATE MODEL FOLDER
# =========================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =========================
# CALLBACKS
# =========================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]


# =========================
# TRAIN
# =========================

print("\nStarting Autoencoder training...\n")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================
# EVALUATION
# =========================

loss, mae = model.evaluate(
    test_generator,
    verbose=1
)

print("\n==============================")
print("TRAINING COMPLETED")
print("==============================")

print("Test Loss :", loss)
print("Test MAE  :", mae)

print("\nModel saved at:")
print(MODEL_PATH)


# =========================
# SAVE TRAINING HISTORY
# =========================

np.save(
    "outputs/train_loss.npy",
    np.array(history.history["loss"])
)

np.save(
    "outputs/val_loss.npy",
    np.array(history.history["val_loss"])
)

print("\nTraining history saved.")
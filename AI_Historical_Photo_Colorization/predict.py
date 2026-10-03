import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = "models/colorization_autoencoder.keras"
IMG_SIZE = 128
OUTPUT_PATH = "outputs/colorized_result.png"


# =========================
# LOAD MODEL
# =========================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# =========================
# GET IMAGE PATH
# =========================

image_path = input(
    "\nEnter the full path of the black-and-white image: "
).strip().strip('"')


# =========================
# CHECK IMAGE
# =========================

if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"\nImage not found:\n{image_path}"
    )


image = cv2.imread(image_path)

if image is None:
    raise ValueError(
        "The selected file could not be read as an image."
    )


# =========================
# PREPROCESS IMAGE
# =========================

original = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

resized = cv2.resize(
    original,
    (IMG_SIZE, IMG_SIZE)
)

gray = cv2.cvtColor(
    resized,
    cv2.COLOR_RGB2GRAY
)

gray_input = gray.astype(
    np.float32
) / 255.0

gray_input = np.expand_dims(
    gray_input,
    axis=(0, -1)
)


# =========================
# COLORIZE
# =========================

print("\nGenerating colorized image...")

prediction = model.predict(
    gray_input,
    verbose=0
)

colorized = prediction[0]

colorized = np.clip(
    colorized,
    0,
    1
)


# =========================
# SAVE RESULT
# =========================

os.makedirs(
    "outputs",
    exist_ok=True
)

plt.imsave(
    OUTPUT_PATH,
    colorized
)


# =========================
# DISPLAY
# =========================

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(resized)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(colorized)
plt.title("AI Colorized")
plt.axis("off")

plt.tight_layout()
plt.show()


print("\n==============================")
print("COLORIZATION COMPLETED")
print("==============================")

print(
    f"Result saved to:\n{OUTPUT_PATH}"
)
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Load the already trained model
model = load_model("cat_dog_mobilenetv2.keras")

print("Trained model loaded successfully!")
print("Starting camera...")
print("Press Q to exit.")


# Open camera
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break


    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Resize image
    image = cv2.resize(
        rgb_frame,
        (224, 224)
    )


    # Normalize
    image = image.astype("float32") / 255.0


    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )


    # Prediction
    prediction = model.predict(
        image,
        verbose=0
    )[0][0]


    # CAT / DOG
    if prediction >= 0.5:

        label = "DOG"
        confidence = prediction * 100

    else:

        label = "CAT"
        confidence = (1 - prediction) * 100


    # Display result
    text = f"{label} - {confidence:.1f}%"


    cv2.rectangle(
        frame,
        (10, 10),
        (420, 80),
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


    # Show camera
    cv2.imshow(
        "Real-Time Cat and Dog Classification",
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Close camera
cap.release()
cv2.destroyAllWindows()

print("Camera closed.")
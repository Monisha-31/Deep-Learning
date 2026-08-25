import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. LOAD CNN MODEL
# ============================================================

MODEL_PATH = "hand_gesture_model.keras"
CLASS_PATH = "gesture_classes.json"
HAND_MODEL = "hand_landmarker.task"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

print("CNN model loaded successfully.")
print("Classes:", class_names)


# ============================================================
# 2. LOAD MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# 3. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("\n================================")
print("HAND GESTURE VIRTUAL ASSISTANT")
print("================================")
print("Camera started.")
print("Press Q to exit.")


timestamp = 0


# ============================================================
# 4. MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read camera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 33

    # Detect hand
    results = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # 5. DEFAULT RESULT
    # ========================================================

    gesture_name = "No Hand"
    confidence = 0.0


    # ========================================================
    # 6. HAND DETECTED
    # ========================================================

    if results.hand_landmarks:

        hand_landmarks = results.hand_landmarks[0]


        # ----------------------------------------------------
        # Find bounding box
        # ----------------------------------------------------

        x_coordinates = [
            landmark.x for landmark in hand_landmarks
        ]

        y_coordinates = [
            landmark.y for landmark in hand_landmarks
        ]

        x_min = max(
            0,
            int(min(x_coordinates) * w) - 30
        )

        y_min = max(
            0,
            int(min(y_coordinates) * h) - 30
        )

        x_max = min(
            w,
            int(max(x_coordinates) * w) + 30
        )

        y_max = min(
            h,
            int(max(y_coordinates) * h) + 30
        )


        # ----------------------------------------------------
        # Crop hand
        # ----------------------------------------------------

        hand_crop = frame[
            y_min:y_max,
            x_min:x_max
        ]


        if hand_crop.size != 0:

            # Resize
            resized = cv2.resize(
                hand_crop,
                (128, 128)
            )

            # Normalize
            image = resized.astype(
                np.float32
            ) / 255.0

            # Add batch dimension
            image = np.expand_dims(
                image,
                axis=0
            )


            # ------------------------------------------------
            # CNN prediction
            # ------------------------------------------------

            predictions = model.predict(
                image,
                verbose=0
            )[0]

            class_index = np.argmax(
                predictions
            )

            confidence = float(
                predictions[class_index]
            )

            gesture_name = class_names[
                class_index
            ]


        # ----------------------------------------------------
        # Draw hand landmarks
        # ----------------------------------------------------

        for landmark in hand_landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )


    # ========================================================
    # 7. DISPLAY RESULT
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (520, 90),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "Gesture: " + gesture_name,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "Confidence: "
        + str(round(confidence * 100, 1))
        + "%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Hand Gesture Virtual Assistant",
        frame
    )


    # ========================================================
    # 8. EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 9. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("Assistant stopped.")
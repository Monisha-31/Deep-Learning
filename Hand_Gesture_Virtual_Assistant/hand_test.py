import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# MediaPipe Hand Landmarker
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# Gesture Recognition Function
# -----------------------------

def recognize_gesture(hand):

    # Finger landmark IDs
    # Thumb: 4
    # Index: 8
    # Middle: 12
    # Ring: 16
    # Pinky: 20

    thumb_tip = hand[4]
    thumb_ip = hand[3]

    index_tip = hand[8]
    index_pip = hand[6]

    middle_tip = hand[12]
    middle_pip = hand[10]

    ring_tip = hand[16]
    ring_pip = hand[14]

    pinky_tip = hand[20]
    pinky_pip = hand[18]


    # -----------------------------
    # Check which fingers are open
    # -----------------------------

    index_open = index_tip.y < index_pip.y
    middle_open = middle_tip.y < middle_pip.y
    ring_open = ring_tip.y < ring_pip.y
    pinky_open = pinky_tip.y < pinky_pip.y


    # -----------------------------
    # Thumb detection
    # -----------------------------

    thumb_open = abs(thumb_tip.x - thumb_ip.x) > 0.05


    # -----------------------------
    # Gesture Rules
    # -----------------------------

    # ✋ Open Palm
    if (
        index_open
        and middle_open
        and ring_open
        and pinky_open
    ):
        return "OPEN PALM"

    # ✊ Fist
    elif (
        not index_open
        and not middle_open
        and not ring_open
        and not pinky_open
    ):
        return "FIST"

    # ✌️ Peace
    elif (
        index_open
        and middle_open
        and not ring_open
        and not pinky_open
    ):
        return "PEACE"

    # ☝️ One Finger
    elif (
        index_open
        and not middle_open
        and not ring_open
        and not pinky_open
    ):
        return "ONE"

    # 👍 Thumbs Up
    elif (
        thumb_open
        and not index_open
        and not middle_open
        and not ring_open
        and not pinky_open
        and thumb_tip.y < thumb_ip.y
    ):
        return "THUMBS UP"

    # 👎 Thumbs Down
    elif (
        thumb_open
        and not index_open
        and not middle_open
        and not ring_open
        and not pinky_open
        and thumb_tip.y > thumb_ip.y
    ):
        return "THUMBS DOWN"

    return "UNKNOWN"


# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

print("Hand Gesture Virtual Assistant Started")
print("Press Q to exit.")

timestamp = 0


# -----------------------------
# Main Loop
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read camera.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 33

    results = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    gesture = "NO HAND"


    # -----------------------------
    # Detect Gesture
    # -----------------------------

    if results.hand_landmarks:

        hand = results.hand_landmarks[0]

        gesture = recognize_gesture(hand)


        # -----------------------------
        # Draw Landmarks
        # -----------------------------

        h, w, _ = frame.shape

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # -----------------------------
        # Draw Connections
        # -----------------------------

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9),
            (9, 13),
            (13, 17),
            (0, 17)
        ]

        for start, end in connections:

            x1 = int(hand[start].x * w)
            y1 = int(hand[start].y * h)

            x2 = int(hand[end].x * w)
            y2 = int(hand[end].y * h)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


    # -----------------------------
    # Display Gesture
    # -----------------------------

    cv2.rectangle(
        frame,
        (10, 60),
        (450, 120),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "Gesture: " + gesture,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "Hand Gesture Virtual Assistant",
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()
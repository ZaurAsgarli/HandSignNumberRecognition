import cv2
import numpy as np
import time
import os
import pyautogui
from ultralytics import YOLO
import tensorflow as tf
from collections import deque

# Load models
yolo_model = YOLO("best_hand_sign.pt")
keras_model = tf.keras.models.load_model("NumberHandSignClassifier.keras")

# Buffer and prediction cooldown
buffer = deque(maxlen=10)
last_prediction_time = 0
prediction_interval = 2  # seconds

# Desktop path
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")


def trigger_action(number):
    if number == 0:
        print("Triggering action for 0: Simulate 'q' key press")
        pyautogui.press("q")
    elif 1 <= number <= 9:
        file_path = os.path.join(desktop_path, f"hand_sign_{number}.txt")
        with open(file_path, "w") as f:
            f.write(f"You have chosen {number}")
        print(f"Created file: {file_path}")
    else:
        print("No action defined for this number")


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = yolo_model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            hand_crop = frame[y1:y2, x1:x2]

            # Draw detection box always
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            current_time = time.time()
            if current_time - last_prediction_time > prediction_interval:
                # Preprocess for Keras model
                hand_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                hand_resized = cv2.resize(hand_gray, (100, 100))
                cv2.imshow("Input to Keras Model", hand_resized)
                hand_input = np.expand_dims(hand_resized, axis=(0, -1))

                # Predict number
                prediction = keras_model.predict(hand_input)
                print(np.round(prediction, 2))  # to see actual probabilities
                predicted_number = np.argmax(prediction)
                confidence = np.max(prediction)
                if confidence > 0.85:
                    buffer.append(predicted_number)
                    if buffer.count(predicted_number) >= 7:
                        trigger_action(predicted_number)
                        buffer.clear()
                        last_prediction_time = current_time  # Update cooldown

                # Display number
                cv2.putText(
                    frame,
                    str(predicted_number),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

    cv2.imshow("Hand Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

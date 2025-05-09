✋🤖 Real-Time Hand Sign Number Recognition
This project is a real-time hand sign number recognition system that detects a hand using a YOLO object detector, classifies the hand sign using a Keras CNN model, and performs specific actions based on the recognized number.

🔍 Features
📸 Live webcam input using OpenCV.

✋ YOLO-based hand detection for localizing hands in real time.

🔢 Keras classifier for predicting hand signs (0–9).

⚡ Action triggering based on the recognized number:

0: Simulates pressing the "q" key.

1–9: Creates a .txt file on the desktop named hand_sign_<number>.txt.

🧠 Models Used
best_hand_sign.pt: YOLO model for hand detection.

NumberHandSignClassifier.keras: CNN model trained to classify grayscale hand sign images (100x100) into digits 0–9.

📂 Files
model.py: Main script that integrates YOLO detection, Keras prediction, and action mapping.

HandDetection.ipynb: Notebook for testing YOLO hand detection.

NumberHandSignRecognition.ipynb: Notebook for training/testing the hand sign classification model.

🛠 Requirements
Python 3.8+

OpenCV

TensorFlow

Ultralytics YOLO

PyAutoGUI

NumPy

🧪 Function Logic
The prediction runs every 2 seconds and requires 7 consistent predictions in a buffer before triggering the action to ensure stability.

📌 Notes
High accuracy is required (confidence > 0.85) to avoid false triggers.

You can customize the trigger_action() function in model.py to perform different tasks.

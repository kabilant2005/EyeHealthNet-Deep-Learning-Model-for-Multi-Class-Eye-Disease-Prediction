import cv2
import numpy as np
import tensorflow as tf
import time
import os

# ============================
# Load Model + Class Labels
# ============================
model_path = r"C:\Users\Kabilan\OneDrive\Desktop\cnn\results\eye_cnn_model.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError("Trained CNN model not found!")

print("Loading CNN Model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Load Class Labels
label_json = r"C:\Users\Kabilan\OneDrive\Desktop\cnn\results\history.json"
import json
with open(label_json, "r") as f:
    history = json.load(f)

# Extract labels in correct order
labels = list(history.get("accuracy", []))  # fallback if missing

# Better: Load from folder structure (recommended)
dataset_path = r"C:\Users\Kabilan\Downloads\archive (1)\Eye dataset"
labels = sorted(os.listdir(dataset_path))

print("Detected Classes:", labels)

# ============================
# Preprocessing Function
# ============================
def preprocess_frame(frame):
    img = cv2.resize(frame, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ============================
# Realtime Prediction
# ============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera Error!")
    exit()

print("🔥 Realtime CNN Eye Disease Detection Started!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected!")
        break

    # Preprocess
    input_img = preprocess_frame(frame)

    # Predict
    pred = model.predict(input_img)
    class_id = np.argmax(pred)
    confidence = pred[0][class_id] * 100
    label = labels[class_id]

    # Overlay text
    cv2.putText(frame, f"Prediction: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display
    cv2.imshow("CNN Eye Disease Detection", frame)

    # Exit key (Q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

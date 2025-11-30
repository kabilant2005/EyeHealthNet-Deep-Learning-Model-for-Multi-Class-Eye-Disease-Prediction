import cv2
import numpy as np
import joblib

# ============================
#   LOAD TRAINED COMPONENTS
# ============================
mlp_model = joblib.load(r"C:\Users\Kabilan\OneDrive\Desktop\cnn\results\mlp\eye_disease_mlp_model.pkl")
scaler = joblib.load(r"C:\Users\Kabilan\OneDrive\Desktop\cnn\results\mlp\scaler.pkl")
label_encoder = joblib.load(r"C:\Users\Kabilan\OneDrive\Desktop\cnn\results\mlp\label_encoder.pkl")

# ============================
#   PREDICTION FUNCTION
# ============================
def predict_eye(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    flattened = resized.flatten().reshape(1, -1)

    scaled = scaler.transform(flattened)

    pred = mlp_model.predict(scaled)
    label = label_encoder.inverse_transform(pred)[0]

    return label

# ============================
#   REAL‑TIME PREDICTION LOOP
# ============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Real-time Eye Disease Prediction Started...")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    # Predict label
    label = predict_eye(frame)

    # Display result on the frame
    cv2.putText(frame,
                f"Prediction: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Eye Disease Detection - MLP Model", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

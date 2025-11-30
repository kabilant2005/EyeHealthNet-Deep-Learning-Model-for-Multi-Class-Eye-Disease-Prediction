import cv2
import joblib
import numpy as np
from skimage.feature import hog

# ============================================================
#                   LOAD TRAINED MODEL + LABEL ENCODER
# ============================================================
model_path = r"C:\Users\Kabilan\OneDrive\Desktop\cnn\results\retino_svm_model.pkl"
label_encoder_path = r"C:\Users\Kabilan\OneDrive\Desktop\cnn\results\label_encoder.pkl"

model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

print("✔ Model Loaded Successfully!")
print("✔ Starting Realtime Prediction...")

# ============================================================
#                   PREDICTION FUNCTION
# ============================================================
def predict_frame(frame_gray):
    img = cv2.resize(frame_gray, (128, 128))

    features, _ = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )

    features = features.reshape(1, -1)

    pred = model.predict(features)
    label = label_encoder.inverse_transform(pred)

    return label[0]

# ============================================================
#                   REAL-TIME OPENCV LOOP
# ============================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received!")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run prediction
    disease = predict_frame(gray)

    # Display prediction on frame
    cv2.putText(frame, f"Predicted: {disease}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Retino Eye Disease - Realtime Prediction", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import sys
from tensorflow.keras.models import load_model

# ============================
# Load NETV2 model
# ============================
MODEL_PATH = "netv2.h5"

try:
    model = load_model(MODEL_PATH)
    print(f"[INFO] Loaded model: {MODEL_PATH}")
except Exception as e:
    print("[ERROR] Cannot load netv2.h5")
    print(e)
    sys.exit()


# ============================
# Preprocess image
# ============================
def preprocess(img):
    img = cv2.resize(img, (224, 224))        # NetV2 input size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ============================
# Predict single image
# ============================
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Cannot read image:", image_path)
        return

    processed = preprocess(img)
    pred = model.predict(processed)[0]

    class_id = np.argmax(pred)
    confidence = pred[class_id] * 100

    print("=====================================")
    print("Prediction Result:")
    print("Class:", class_id)
    print("Confidence:", f"{confidence:.2f}%")
    print("=====================================")


# ============================
# Realtime webcam prediction
# ============================
def predict_realtime():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("[ERROR] Webcam not found!")
        return

    print("[INFO] Realtime prediction started...")
    print("[INFO] Press 'q' to exit")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        processed = preprocess(frame)
        pred = model.predict(processed)[0]

        class_id = np.argmax(pred)
        confidence = pred[class_id] * 100

        text = f"Class: {class_id}, Conf: {confidence:.2f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.imshow("NetV2 Realtime Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# ============================
# Main Control
# ============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inference_netv2.py <image_path>")
        print("  python inference_netv2.py realtime")
        sys.exit()

    arg = sys.argv[1]

    if arg.lower() == "realtime":
        predict_realtime()
    else:
        predict_image(arg)

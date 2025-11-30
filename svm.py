import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import matplotlib.pyplot as plt

# ============================================================
#                   RESULTS SAVE PATH
# ============================================================
results_path = r"C:\Users\Kabilan\OneDrive\Desktop\cnn\results"
os.makedirs(results_path, exist_ok=True)

# ============================================================
#                   DATASET PATH
# ============================================================
dataset_path = r"C:\Users\Kabilan\Downloads\archive (1)\Eye dataset"

# ============================================================
#         LOAD IMAGES + HOG FEATURE EXTRACTION (RETINO)
# ============================================================
images = []
labels = []

print("Loading Retino eye dataset...")

for disease_class in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, disease_class)

    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):

            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (128, 128))

                # HOG Feature Extractor
                features, _ = hog(
                    img,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                    visualize=True
                )

                images.append(features)
                labels.append(disease_class)

print("Dataset loaded successfully!")

# ============================================================
#               PREPARE TRAINING DATA
# ============================================================
X = np.array(images)
y = np.array(labels)

# Encode disease labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ============================================================
#               TRAIN RETINO EYE SVM MODEL
# ============================================================
print("Training Retino Eye Disease SVM Model...")

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

print("Model training completed!")

# ============================================================
#                   EVALUATION
# ============================================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy : {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy  : {test_accuracy * 100:.2f}%")

# Convert classification report to dict
report_dict = classification_report(
    y_test,
    y_test_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

# ============================================================
#            SAVE CLASSIFICATION REPORT (.TXT)
# ============================================================
report_file = os.path.join(results_path, "retino_classification_report.txt")

with open(report_file, "w") as f:
    f.write(classification_report(
        y_test,
        y_test_pred,
        target_names=label_encoder.classes_
    ))

print(f"Classification report saved to {report_file}")

# ============================================================
#               SAVE MODEL + LABEL ENCODER
# ============================================================
joblib.dump(model, os.path.join(results_path, "retino_svm_model.pkl"))
joblib.dump(label_encoder, os.path.join(results_path, "label_encoder.pkl"))

print("Model + label encoder saved!")

# ============================================================
#               SAVE HISTORY JSON (FULL RESULTS)
# ============================================================
history = {
    "training_accuracy": float(train_accuracy),
    "testing_accuracy": float(test_accuracy),
    "labels": list(label_encoder.classes_),
    "classification_report": report_dict
}

history_file = os.path.join(results_path, "retino_training_history.json")

with open(history_file, "w") as f:
    json.dump(history, f, indent=4)

print(f"Training history saved to {history_file}")

# ============================================================
#                   SAVE ACCURACY GRAPH
# ============================================================
plt.figure(figsize=(7, 4))
plt.bar(["Train Accuracy", "Test Accuracy"], [train_accuracy, test_accuracy])
plt.ylabel("Accuracy")
plt.title("Retino Eye Disease – Train vs Test Accuracy")
plt.grid(True)

graph_file = os.path.join(results_path, "retino_accuracy_graph.png")
plt.savefig(graph_file)
plt.close()

print(f"Accuracy graph saved to {graph_file}")

# ============================================================
#                PREDICT NEW RETINO IMAGE
# ============================================================
def predict_retino_eye(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Invalid Image"

    img = cv2.resize(img, (128, 128))

    features, _ = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )

    features = features.reshape(1, -1)
    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)
    return label[0]

# Example:
# print(predict_retino_eye(r"C:\path\image.jpg"))

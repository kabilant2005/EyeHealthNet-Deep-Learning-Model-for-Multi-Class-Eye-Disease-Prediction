import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import matplotlib.pyplot as plt

# ============================================================
#                   RESULTS SAVE PATH
# ============================================================
results_path = r"C:\Users\Kabilan\OneDrive\Desktop\cnn\results\mlp"
os.makedirs(results_path, exist_ok=True)

# ============================================================
#                   DATASET PATH
# ============================================================
dataset_path = r"C:\Users\Kabilan\Downloads\archive (1)\Eye dataset"

# ============================================================
#               LOAD IMAGES + FLATTEN
# ============================================================
images = []
labels = []

print("Loading dataset...")
for label_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label_folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (64, 64))
                img_flat = img.flatten()
                images.append(img_flat)
                labels.append(label_folder)

print("Dataset loaded!")

# ============================================================
#                   DATA PREPARATION
# ============================================================
X = np.array(images)
y = np.array(labels)

# Label Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale for MLP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# ============================================================
#                   TRAIN MLP CLASSIFIER
# ============================================================
print("Training MLP classifier...")
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    max_iter=50,
    random_state=42,
    verbose=True
)

mlp_clf.fit(X_train, y_train)
print("Training completed!")

# ============================================================
#                       EVALUATION
# ============================================================
y_train_pred = mlp_clf.predict(X_train)
y_test_pred = mlp_clf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy : {test_acc*100:.2f}%")

report_dict = classification_report(
    y_test, y_test_pred, target_names=le.classes_, output_dict=True
)

# ============================================================
#               SAVE CLASSIFICATION REPORT TXT
# ============================================================
report_path = os.path.join(results_path, "classification_report.txt")

with open(report_path, "w") as f:
    f.write(classification_report(y_test, y_test_pred, target_names=le.classes_))

print(f"Classification Report Saved: {report_path}")

# ============================================================
#        SAVE MODEL, SCALER, LABEL ENCODER
# ============================================================
joblib.dump(mlp_clf, os.path.join(results_path, "eye_disease_mlp_model.pkl"))
joblib.dump(scaler, os.path.join(results_path, "scaler.pkl"))
joblib.dump(le, os.path.join(results_path, "label_encoder.pkl"))

print("Model, Scaler, Encoder Saved!")

# ============================================================
#                SAVE TRAINING HISTORY JSON
# ============================================================
history = {
    "train_accuracy": float(train_acc),
    "test_accuracy": float(test_acc),
    "classification_report": report_dict,
    "classes": list(le.classes_)
}

json_path = os.path.join(results_path, "training_history.json")
with open(json_path, "w") as f:
    json.dump(history, f, indent=4)

print(f"Training history saved: {json_path}")

# ============================================================
#                      SAVE ACCURACY GRAPH
# ============================================================
plt.figure(figsize=(6,4))
plt.bar(["Train Accuracy", "Test Accuracy"], [train_acc, test_acc], color=["blue", "orange"])
plt.ylabel("Accuracy")
plt.title("MLP Training vs Testing Accuracy")
plt.grid(True)

graph_path = os.path.join(results_path, "accuracy_graph.png")
plt.savefig(graph_path)
plt.close()

print(f"Accuracy graph saved: {graph_path}")

# ============================================================
#                PREDICTION FUNCTION
# ============================================================
def predict_eye_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Invalid image"

    img = cv2.resize(img, (64, 64))
    img_flat = img.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flat)

    pred = mlp_clf.predict(img_scaled)
    label = le.inverse_transform(pred)
    return label[0]

# Example:
# print(predict_eye_image("path_to_image.jpg"))

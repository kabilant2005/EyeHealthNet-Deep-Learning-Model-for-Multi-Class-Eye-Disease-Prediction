import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# ============================
# Results Save Folder
# ============================
save_path = r"C:\Users\Kabilan\OneDrive\Desktop\cnn\results"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# ============================
# Dataset Path
# ============================
base_path = r"C:\Users\Kabilan\Downloads\archive (1)\Eye dataset"

if not os.path.exists(base_path):
    raise ValueError("Dataset path does not exist. Check the path again.")

print("Classes found:", os.listdir(base_path))

# ============================
# Data Loaders
# ============================
datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    base_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

test_data = datagen.flow_from_directory(
    base_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ============================
# CNN Model
# ============================
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # FIXED INPUT SHAPE WARNING
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================
# Save Model Architecture (JSON)
# ============================
model_json = model.to_json()
with open(os.path.join(save_path, "model_architecture.json"), "w", encoding="utf-8") as f:
    f.write(model_json)

print("Model architecture saved safely.")

# ============================
# Train Model
# ============================
history = model.fit(
    train_data,
    epochs=15,
    validation_data=test_data
)

# ============================
# Save History as JSON
# ============================
with open(os.path.join(save_path, "history.json"), "w") as f:
    json.dump(history.history, f, indent=4)

# ============================
# Save History Pickle
# ============================
with open(os.path.join(save_path, "history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# ============================
# Evaluate Model
# ============================
loss, accuracy = model.evaluate(test_data)

metrics = {
    "test_loss": float(loss),
    "test_accuracy": float(accuracy),
    "num_classes": int(train_data.num_classes),
    "training_samples": int(train_data.samples),
    "validation_samples": int(test_data.samples)
}

with open(os.path.join(save_path, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# ============================
# Save Trained Model
# ============================
model.save(os.path.join(save_path, "eye_cnn_model.h5"))

# ============================
# Accuracy Graph
# ============================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.grid(True)
plt.savefig(os.path.join(save_path, "accuracy_graph.png"))
plt.close()

# ============================
# Loss Graph
# ============================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.savefig(os.path.join(save_path, "loss_graph.png"))
plt.close()

# ============================
# Predictions
# ============================
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes
labels = list(test_data.class_indices.keys())

# ============================
# Classification Report
# ============================
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

with open(os.path.join(save_path, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

# ============================
# Confusion Matrix
# ============================
cm = confusion_matrix(y_true, y_pred)

with open(os.path.join(save_path, "confusion_matrix.json"), "w") as f:
    json.dump(cm.tolist(), f, indent=4)

# ============================
# Predictions CSV
# ============================
df = pd.DataFrame({
    "true_label": [labels[i] for i in y_true],
    "predicted_label": [labels[i] for i in y_pred]
})
df.to_csv(os.path.join(save_path, "predictions.csv"), index=False)

print("\nAll results saved successfully without Unicode errors!")

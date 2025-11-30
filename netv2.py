"""
EfficientNetV2 training + full results pipeline + inference + realtime OpenCV
Save this file as:
C:\\Users\\Kabilan\\OneDrive\\Desktop\\cnn\\netv2\\netv2_pipeline.py

Usage:
    python netv2_pipeline.py train
    python netv2_pipeline.py predict "C:\\path\\to\\image.jpg"
    python netv2_pipeline.py realtime
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import cv2

# ===============================
# PATHS
# ===============================
BASE_DIR = r"C:\Users\Kabilan\OneDrive\Desktop\cnn\netv2"
os.makedirs(BASE_DIR, exist_ok=True)

DATASET_PATH = r"C:\Users\Kabilan\Downloads\archive (1)\Eye dataset"
RESULTS_PATH = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42


# ===============================
# UTILS
# ===============================
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def plot_and_save(history, out_dir):
    # accuracy
    plt.figure(figsize=(8,5))
    plt.plot(history.get("accuracy", []), label="train_acc")
    plt.plot(history.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_plot.png"))
    plt.close()

    # loss
    plt.figure(figsize=(8,5))
    plt.plot(history.get("loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_plot.png"))
    plt.close()


# ===============================
# DATA LOADER
# ===============================
def make_data_generators(dataset_path, img_size, batch_size):
    datagen = ImageDataGenerator(
        rescale=1/255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED
    )

    save_json(train_gen.class_indices, os.path.join(RESULTS_PATH, "class_indices.json"))
    return train_gen, val_gen


# ===============================
# MODEL
# ===============================
def build_model(num_classes, img_size):
    try:
        base = tf.keras.applications.EfficientNetV2B0(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet"
        )
    except:
        base = tf.keras.applications.EfficientNetV2S(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet"
        )

    base.trainable = False

    inputs = layers.Input((img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ===============================
# TRAINING PIPELINE
# ===============================
def train_and_save():
    print("Loading dataset...")
    train_gen, val_gen = make_data_generators(DATASET_PATH, IMG_SIZE, BATCH_SIZE)
    num_classes = train_gen.num_classes

    model = build_model(num_classes, IMG_SIZE)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join(RESULTS_PATH, f"netv2_best_{ts}.h5")

    cb = [
        callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        callbacks.CSVLogger(os.path.join(RESULTS_PATH, f"training_log_{ts}.csv"))
    ]

    history1 = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=cb)

    # fine tuning
    model.layers[1].trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    history2 = model.fit(train_gen, epochs=3, validation_data=val_gen)

    # merge history
    history = {}
    keys = set(history1.history.keys()).union(history2.history.keys())
    for k in keys:
        history[k] = history1.history.get(k, []) + history2.history.get(k, [])

    # save
    model.save(os.path.join(RESULTS_PATH, "efficientnetv2_final.h5"))
    save_json(history, os.path.join(RESULTS_PATH, "history.json"))
    save_pickle(history, os.path.join(RESULTS_PATH, "history.pkl"))
    plot_and_save(history, RESULTS_PATH)

    # validation reports
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    labels = {v: k for k, v in val_gen.class_indices.items()}
    label_names = [labels[i] for i in range(len(labels))]

    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    save_json(report, os.path.join(RESULTS_PATH, "classification_report.json"))

    cm = confusion_matrix(y_true, y_pred)
    save_json(cm.tolist(), os.path.join(RESULTS_PATH, "confusion_matrix.json"))

    # accuracy print
    loss, acc = model.evaluate(val_gen)
    print("\n=========== RESULTS ===========")
    print("Validation Accuracy:", round(acc * 100, 2), "%")
    print("================================\n")


# ===============================
# INFERENCE
# ===============================
def load_model_for_inference():
    model_path = os.path.join(RESULTS_PATH, "efficientnetv2_final.h5")
    model = tf.keras.models.load_model(model_path)

    with open(os.path.join(RESULTS_PATH, "class_indices.json"), "r") as f:
        ci = json.load(f)

    inv = {v: k for k, v in ci.items()}
    return model, inv


def preprocess_bgr(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(img.astype("float32") / 255.0, axis=0)


def predict_image(path):
    model, inv = load_model_for_inference()
    img = cv2.imread(path)
    if img is None:
        print("Invalid image path")
        return

    pred = model.predict(preprocess_bgr(img))[0]
    idx = int(np.argmax(pred))
    print(f"Prediction: {inv[idx]}   Confidence: {np.max(pred)*100:.2f}%")


# ===============================
# REALTIME
# ===============================
def realtime_opencv():
    model, inv = load_model_for_inference()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam error")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred = model.predict(preprocess_bgr(frame))[0]
        idx = int(np.argmax(pred))
        label = inv[idx]
        conf = np.max(pred) * 100

        cv2.putText(frame, f"{label} ({conf:.1f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("EfficientNetV2 Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python netv2_pipeline.py train")
        print("  python netv2_pipeline.py predict <image>")
        print("  python netv2_pipeline.py realtime")
        sys.exit()

    cmd = sys.argv[1].lower()

    if cmd == "train":
        train_and_save()
    elif cmd == "predict":
        predict_image(sys.argv[2])
    elif cmd == "realtime":
        realtime_opencv()
    else:
        print("Invalid command")

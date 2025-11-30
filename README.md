# EyeHealthNet – Personalized Multi-Class Eye Disease Prediction

## 💡 Project Concept
EyeHealthNet is a personal deep learning project designed to **detect and classify multiple eye diseases** from images or feature datasets.  
The goal is to provide **an intelligent tool for early eye disorder detection**, assisting both medical professionals and users in understanding eye health.  
This project demonstrates my ability to combine **CNN, SVM, and MLP models** in a unified system for image and feature-based classification.

---

## 🛠 Method / Approach

1. **Data Collection & Preprocessing**  
   - Gathered eye images and structured feature datasets.  
   - Preprocessing steps include resizing, normalization, and label encoding.  
   - Ensures consistent input for all models.

2. **Model Training**  
   - **CNN**: For direct image-based disease classification.  
   - **SVM**: Uses extracted features for classification.  
   - **MLP**: Multi-Layer Perceptron for feature-based prediction.  
   - All models are trained, validated, and saved for inference.

3. **Prediction / Inference**  
   - Load any trained model.  
   - Input an image or feature set → output predicted disease + confidence score.  
   - Designed for real-time or batch processing of eye images.

4. **Results & Evaluation**  
   - Accuracy, precision, recall, and F1-score are calculated for each model.  
   - Confusion matrices and performance graphs are saved in the `results/` folder.  
   - Enables comparison between CNN, SVM, and MLP models.

---

## 🔧 Setup & Run Instructions

### 1️⃣ Install Required Packages

EyeHealthNet/
│── cnn.py
│── cnninference.py
│── mlp.py
│── mlpinference.py
│── svm.py
│── svminference.py
│── models/
│── results/
│── label_encoder.pkl
│── scaler.pkl
│── eye_cnn_model.h5
│── eye_disease_svm_model.pkl
│── eye_disease_mlp_model.pkl
│── README.md
│── .gitignore

python cnn.py 
python mlp.py
python cnninference.py --image sample.jpg
python svminference.py
python mlpinference.py
```bash
pip install tensorflow scikit-learn opencv-python numpy matplotlib jobli


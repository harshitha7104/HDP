# ❤️ Heart Disease Predictor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xJSWtuuIxKvq5-ZcQG7btZZC0L8tt6XU#scrollTo=qlOyZdxdSow7)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Network-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)

> **Live Demo (Flask)** 

---

## 📌 Overview

A deep learning web application that predicts the likelihood of heart disease based on 11 patient health inputs. The model is a **feedforward neural network** built with PyTorch, trained on 918 patient records.

The project ships with **two frontends**:
- A **Streamlit app** for quick local use
- A **Flask web app** for deployment

---

## 🤖 Model & Results

A **3-hidden-layer feedforward neural network** trained on an 80/20 train/test split (734 train, 184 test):

### Architecture

```
Input (20 features)
    ↓
Linear(20 → 64) + BatchNorm + ReLU + Dropout(30%)
    ↓
Linear(64 → 32) + BatchNorm + ReLU + Dropout(20%)
    ↓
Linear(32 → 16) + ReLU
    ↓
Linear(16 → 1) + Sigmoid  →  probability (0–1)
```

### Performance

| Metric | Score |
|---|---|
| **Accuracy** | **89.67%** |
| Precision | 90% |
| Recall | 90% |
| F1 Score | 90% |

**Confusion Matrix:**

```
                  Predicted: No Disease   Predicted: Disease
Actual: No Disease        70                    11
Actual: Disease            8                    95
```

---

## 📂 Dataset

**918 patient records** from a combination of 5 public heart disease datasets (Cleveland, Hungarian, Switzerland, Long Beach VA, Stalog).

| Feature | Description |
|---|---|
| Age | Age in years |
| Sex | Male / Female |
| ChestPainType | ASY / ATA / NAP / TA |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl (0/1) |
| RestingECG | Normal / ST / LVH |
| MaxHR | Maximum heart rate during exercise |
| ExerciseAngina | Exercise-induced chest pain (Y/N) |
| Oldpeak | ST depression during exercise |
| ST_Slope | Up / Flat / Down |
| **HeartDisease** | **Target — 0 = No Disease, 1 = Disease** |

---

## ⚙️ How It Works

### 1. Preprocessing
The 5 categorical columns are **one-hot encoded** using `pd.get_dummies()`, expanding the feature space from 12 → **20 columns**:

```
Sex="M"             →  Sex_M=1, Sex_F=0
ChestPainType="ASY" →  ChestPainType_ASY=1, others=0
...
```

### 2. Feature Scaling
All 20 features are normalised using `StandardScaler` (mean=0, std=1). This is critical for neural networks — without it, features with large values (e.g. `RestingBP=140`) would dominate over small ones (e.g. `Oldpeak=1.2`).

### 3. Training (`run_model.py`)
The neural network is trained for 300 epochs using:
- **Loss:** Binary Cross Entropy (`BCELoss`)
- **Optimiser:** Adam (`lr=1e-3`, `weight_decay=1e-4`)
- **Scheduler:** StepLR — halves learning rate every 100 epochs

### 4. Inference
The saved `nn_model.pkl` and `scaler.pkl` are loaded at runtime. User inputs are preprocessed into the same 20-feature format, scaled, converted to a PyTorch tensor, and passed through the network. Output is a probability (0–1); values above **0.5** are classified as High Risk.

---

## 🗂️ Project Structure

```
├── heart.csv                  # Raw dataset (918 records)
├── run_model.py               # Train neural network, evaluate, save model
├── train_nn.py                # Standalone neural network trainer
├── nn_model.pkl               # Saved PyTorch neural network
├── scaler.pkl                 # Saved StandardScaler (required at inference)
├── streamlit_app.py           # Streamlit frontend
├── app.py                     # Flask API backend
├── templates/
│   └── index.html             # Flask HTML frontend
├── heart_disease_predictor.ipynb  # Jupyter notebook (full walkthrough)
├── requirements.txt
└── procfile                   # Deployment config (Render)
```

---

## 🚀 Running Locally

### Step 1 — Train the Neural Network
```bash
pip install -r requirements.txt
python run_model.py
```
This trains the model and saves `nn_model.pkl` + `scaler.pkl`.

### Step 2 — Launch the App

**Streamlit:**
```bash
streamlit run streamlit_app.py
```

**Flask:**
```bash
python app.py
```
Then open http://localhost:5000

---

## 📦 Dependencies

```
flask
flask-cors
torch
scikit-learn
pandas
numpy
matplotlib
streamlit
gunicorn
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📋 Input Fields

| Field | Type | Example |
|---|---|---|
| Age | Number | 54 |
| Sex | Male / Female | Male |
| Chest Pain Type | 4 options | Asymptomatic |
| Resting Blood Pressure | Number (mm Hg) | 130 |
| Cholesterol | Number (mg/dl) | 245 |
| Fasting Blood Sugar > 120 | Checkbox | No |
| Resting ECG | 3 options | Normal |
| Max Heart Rate | Number (bpm) | 150 |
| Exercise Angina | Yes / No | No |
| ST Depression (Oldpeak) | Slider (-3 to 7) | 1.2 |
| ST Slope | 3 options | Flat |

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider regarding any medical concerns.

# ❤️ Heart Disease Predictor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xJSWtuuIxKvq5-ZcQG7btZZC0L8tt6XU#scrollTo=qlOyZdxdSow7)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)

> **Live Demo (Flask)** → https://heart-disease-predictor-ft20.onrender.com/

---

## 📌 Overview

A machine learning web application that predicts the likelihood of heart disease based on 11 patient health inputs. Three models were trained and compared — the best performing model (Random Forest) is used for analysis, while the deployed app uses a pre-trained XGBoost model.

The project ships with **two frontends**:
- A **Streamlit app** for quick local use
- A **Flask web app** for deployment

---

## 🤖 Models & Results

Three classifiers were trained on an 80/20 train/test split (734 train, 184 test):

| Model | Accuracy |
|---|---|
| Decision Tree (max_depth=3) | 86.41% |
| XGBoost | 86.96% |
| **Random Forest** | **89.13%** ✅ Best |

**XGBoost Confusion Matrix** (deployed model):

```
                  Predicted: No Disease   Predicted: Disease
Actual: No Disease        72                     9
Actual: Disease           15                    88
```

| Metric | Score |
|---|---|
| Precision | 91% |
| Recall | 85% |
| F1 Score | 88% |
| Overall Accuracy | 87% |

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
Sex="M"            →  Sex_M=1, Sex_F=0
ChestPainType="ASY" →  ChestPainType_ASY=1, others=0
...
```

### 2. Training (`run_model.py`)
All three models are trained and evaluated. Results and a confusion matrix are printed to the console.

### 3. Inference
The saved `xgboost_model.pkl` is loaded at runtime. User inputs are preprocessed into the same 20-feature format and passed to `xgb.DMatrix` for prediction. Output is a probability (0–1); values above **0.5** are classified as High Risk.

---

## 🗂️ Project Structure

```
├── heart.csv                  # Raw dataset (918 records)
├── run_model.py               # Train & evaluate all 3 models
├── xgboost_model.pkl          # Pre-trained XGBoost model
├── streamlit_app.py           # Streamlit frontend
├── app.py                     # Flask API backend
├── templates/
│   └── index.html             # Flask HTML frontend
├── requirements.txt
└── procfile                   # Deployment config (Render)
```

---

## 🚀 Running Locally

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Flask App
```bash
pip install -r requirements.txt
python app.py
```
Then open http://localhost:5000

### Train Models from Scratch
```bash
python run_model.py
```

---

## 📦 Dependencies

```
flask
flask-cors
xgboost
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

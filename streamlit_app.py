import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ── Neural Network definition (must match train_nn.py exactly) ────────────────
class HeartNet(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    with open("nn_model.pkl", "rb") as f:
        model = pickle.load(f)
    model.eval()
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

FEATURE_NAMES = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Sex_F", "Sex_M",
    "ChestPainType_ASY", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "RestingECG_LVH", "RestingECG_Normal", "RestingECG_ST",
    "ExerciseAngina_N", "ExerciseAngina_Y",
    "ST_Slope_Down", "ST_Slope_Flat", "ST_Slope_Up",
]

SEX_MAP = {"Male": "M", "Female": "F"}

CHEST_PAIN_MAP = {
    "No chest pain at all (Asymptomatic)": "ASY",
    "Unusual or mild chest discomfort (Atypical Angina)": "ATA",
    "Chest pain unrelated to the heart (Non-Anginal)": "NAP",
    "Classic chest tightness or pressure (Typical Angina)": "TA",
}

ECG_MAP = {
    "Normal": "Normal",
    "Irregular heart signal pattern (ST-T wave abnormality)": "ST",
    "Thickened heart muscle detected (Left Ventricular Hypertrophy)": "LVH",
}

ANGINA_MAP = {"No": "N", "Yes": "Y"}

ST_SLOPE_MAP = {
    "Rising — normal healthy response (Upsloping)": "Up",
    "Flat — borderline, needs attention": "Flat",
    "Falling — abnormal, higher risk (Downsloping)": "Down",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Risk Predictor")
st.caption("Powered by a Neural Network (PyTorch) · For educational purposes only")
st.divider()

with st.form("prediction_form"):

    # Section 1 — Personal Details
    st.subheader("🧍 Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=50,
            help="Your current age in years."
        )
    with col2:
        sex_label = st.selectbox("Biological Sex", list(SEX_MAP.keys()))

    st.divider()

    # Section 2 — Symptoms
    st.subheader("🤕 Symptoms")
    col3, col4 = st.columns(2)
    with col3:
        chest_pain_label = st.selectbox(
            "How would you describe your chest pain?",
            list(CHEST_PAIN_MAP.keys()),
            help="Pick the option that best matches what you typically feel."
        )
    with col4:
        exercise_angina_label = st.selectbox(
            "Do you get chest pain or tightness during exercise?",
            list(ANGINA_MAP.keys()),
            help="Exercise-induced angina means chest discomfort that appears during physical activity."
        )

    st.divider()

    # Section 3 — Vitals & Lab Results
    st.subheader("🔬 Vitals & Lab Results")
    col5, col6 = st.columns(2)
    with col5:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120,
            help="Your blood pressure measured while at rest. Normal is below 120 mm Hg."
        )
        cholesterol = st.number_input(
            "Cholesterol Level (mg/dl)", min_value=0, max_value=700, value=200,
            help="Total cholesterol in your blood. Healthy: below 200 · Borderline: 200-239 · High: 240+."
        )
        fasting_bs = st.checkbox(
            "My fasting blood sugar is above 120 mg/dl",
            help="Fasting means no food for 8+ hours. Blood sugar above 120 mg/dl may indicate diabetes."
        )
    with col6:
        max_hr = st.number_input(
            "Highest Heart Rate During Exercise (bpm)", min_value=60, max_value=250, value=150,
            help="The peak heart rate you reached during a physical stress test. Typical range: 100-190 bpm."
        )
        oldpeak = st.slider(
            "Heart Stress Level During Exercise (ST Depression)",
            min_value=-3.0, max_value=7.0, value=0.0, step=0.1,
            help="Measures how much the heart's electrical signal drops during exercise vs. rest. 0 = no stress."
        )

    st.divider()

    # Section 4 — ECG Results
    st.subheader("📋 Heart Tracing (ECG) Results")
    col7, col8 = st.columns(2)
    with col7:
        resting_ecg_label = st.selectbox(
            "What did your resting ECG show?",
            list(ECG_MAP.keys()),
            help="An ECG records the electrical activity of your heart while you rest."
        )
    with col8:
        st_slope_label = st.selectbox(
            "How did your heart signal change during exercise?",
            list(ST_SLOPE_MAP.keys()),
            help="A rising signal is healthy; flat or falling signals may indicate reduced blood flow."
        )

    submitted = st.form_submit_button(
        "🔍 Check My Heart Disease Risk", use_container_width=True, type="primary"
    )

# ── Inference ─────────────────────────────────────────────────────────────────
if submitted:
    sex             = SEX_MAP[sex_label]
    chest_pain      = CHEST_PAIN_MAP[chest_pain_label]
    resting_ecg     = ECG_MAP[resting_ecg_label]
    exercise_angina = ANGINA_MAP[exercise_angina_label]
    st_slope        = ST_SLOPE_MAP[st_slope_label]

    # Build the 20-feature row (same as before)
    row = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)
    row["Age"]         = age
    row["RestingBP"]   = resting_bp
    row["Cholesterol"] = cholesterol
    row["FastingBS"]   = int(fasting_bs)
    row["MaxHR"]       = max_hr
    row["Oldpeak"]     = oldpeak
    row[f"Sex_{sex}"]                        = 1
    row[f"ChestPainType_{chest_pain}"]       = 1
    row[f"RestingECG_{resting_ecg}"]         = 1
    row[f"ExerciseAngina_{exercise_angina}"] = 1
    row[f"ST_Slope_{st_slope}"]              = 1

    # Scale → tensor → predict
    scaled = scaler.transform(row.values.astype(np.float32))
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = float(model(tensor).item())

    prediction = int(prob > 0.5)

    st.divider()
    st.subheader("Your Result")

    if prediction == 1:
        st.error("### 🔴 High Risk of Heart Disease")
        st.write("The neural network detected patterns commonly associated with heart disease. **Please consult a doctor as soon as possible.**")
    else:
        st.success("### 🟢 Low Risk of Heart Disease")
        st.write("The neural network found no strong indicators of heart disease based on the data you provided.")

    pct = prob * 100
    st.metric(
        label="Estimated Risk Probability",
        value=f"{pct:.1f}%",
        delta=f"{'Above' if prob > 0.5 else 'Below'} the 50% risk threshold"
    )
    st.progress(prob)

    with st.expander("What does this probability mean?"):
        st.write(
            "The neural network outputs a score between 0% and 100%. "
            "Scores **above 50%** are classified as high risk. "
            "This is based on patterns learned from 918 patient records — "
            "it is not a medical diagnosis."
        )

    st.caption(
        "This tool is for educational purposes only and is not a substitute for "
        "professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider."
    )

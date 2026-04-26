from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

app = Flask(__name__)
CORS(app)

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

# ── Load model + scaler ───────────────────────────────────────────────────────
try:
    with open('nn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    model.eval()
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Neural network model loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}. Run train_nn.py first.")
    model = None
    scaler = None

FEATURE_NAMES = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Sex_F', 'Sex_M',
    'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_N', 'ExerciseAngina_Y',
    'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
]

def preprocess_input(data):
    row = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)
    row['Age']        = data['age']
    row['RestingBP']  = data['resting_bp']
    row['Cholesterol']= data['cholesterol']
    row['FastingBS']  = 1 if data['fasting_bs'] else 0
    row['MaxHR']      = data['max_hr']
    row['Oldpeak']    = data['oldpeak']
    row[f"Sex_{data['sex']}"]                          = 1
    row[f"ChestPainType_{data['chest_pain_type']}"]    = 1
    row[f"RestingECG_{data['resting_ecg']}"]           = 1
    row[f"ExerciseAngina_{data['exercise_angina']}"]   = 1
    row[f"ST_Slope_{data['st_slope']}"]                = 1
    return row

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Run train_nn.py first.'}), 500

    try:
        data = request.json

        required_fields = [
            'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
            'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina', 'oldpeak', 'st_slope'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Preprocess → scale → tensor → predict
        processed = preprocess_input(data)
        scaled     = scaler.transform(processed.values.astype(np.float32))
        tensor     = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            prob = float(model(tensor).item())

        prediction = int(prob > 0.5)

        return jsonify({
            'prediction': prediction,
            'probability': prob,
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'confidence': float(max(prob, 1 - prob))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, host='0.0.0.0', port=5000)

"""
train_nn.py
Trains a PyTorch neural network on heart.csv and saves:
  - nn_model.pkl   (the trained PyTorch model)
  - scaler.pkl     (the StandardScaler used to normalise inputs)
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 55
torch.manual_seed(RANDOM_STATE)

# ── 1. Load & preprocess (identical to existing pipeline) ─────────────────────
df = pd.read_csv("heart.csv")
cat_variables = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
df = pd.get_dummies(data=df, prefix=cat_variables, columns=cat_variables)

FEATURE_NAMES = [x for x in df.columns if x != "HeartDisease"]
X = df[FEATURE_NAMES].values.astype(np.float32)
y = df["HeartDisease"].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, random_state=RANDOM_STATE
)

# ── 2. Scale features (critical for neural networks) ─────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

# ── 3. Define the neural network ──────────────────────────────────────────────
class HeartNet(nn.Module):
    """
    3-hidden-layer feedforward network:
      Input(20) → 64 → 32 → 16 → Output(1)
    BatchNorm + Dropout for regularisation.
    """
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
            nn.Sigmoid(),          # output = probability of heart disease
        )

    def forward(self, x):
        return self.net(x)

model = HeartNet(input_dim=X_train.shape[1])

# ── 4. Training ───────────────────────────────────────────────────────────────
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

EPOCHS = 300
print(f"Training for {EPOCHS} epochs...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t)
    loss  = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss  = criterion(val_preds, y_val_t).item()
            val_acc   = accuracy_score(y_val, (val_preds.numpy() > 0.5).astype(int))
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ── 5. Final evaluation ───────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    val_probs = model(X_val_t).numpy().flatten()
    val_preds = (val_probs > 0.5).astype(int)

print("\n" + "=" * 50)
print("NEURAL NETWORK FINAL RESULTS")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
print()
print(classification_report(y_val, val_preds, target_names=["No Disease", "Disease"]))

cm = confusion_matrix(y_val, val_preds)
print("Confusion Matrix")
print("=" * 50)
print(f"                Predicted No  Predicted Yes")
print(f"Actual No       {cm[0][0]:^13} {cm[0][1]:^13}")
print(f"Actual Yes      {cm[1][0]:^13} {cm[1][1]:^13}")

# ── 6. Save model + scaler ────────────────────────────────────────────────────
with open("nn_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Saved: nn_model.pkl and scaler.pkl")

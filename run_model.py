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

# ── 1. Load & preprocess ──────────────────────────────────────────────────────
df = pd.read_csv('heart.csv')
cat_variables = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df = pd.get_dummies(data=df, prefix=cat_variables, columns=cat_variables)

features = [x for x in df.columns if x != 'HeartDisease']
X = df[features].values.astype(np.float32)
y = df['HeartDisease'].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, random_state=RANDOM_STATE
)
print(f"Train samples: {len(X_train)} | Test samples: {len(X_val)}\n")

# ── 2. Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

# ── 3. Neural Network ─────────────────────────────────────────────────────────
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

model     = HeartNet(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# ── 4. Training loop ──────────────────────────────────────────────────────────
EPOCHS = 300
print(f"Training Neural Network for {EPOCHS} epochs...\n")

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
print("NEURAL NETWORK ACCURACY ON TEST DATA")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_val, val_preds, target_names=["No Disease", "Disease"]))

print("CONFUSION MATRIX")
print("=" * 50)
cm = confusion_matrix(y_val, val_preds)
print(f"                Predicted No  Predicted Yes")
print(f"Actual No       {cm[0][0]:^13} {cm[0][1]:^13}")
print(f"Actual Yes      {cm[1][0]:^13} {cm[1][1]:^13}")

# ── 6. Per-sample predictions (first 20) ─────────────────────────────────────
print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS ON TEST DATA (first 20 rows)")
print("=" * 50)
results = pd.DataFrame({'Actual': y_val, 'NN_Pred': val_preds})
results['Correct'] = (results['Actual'] == results['NN_Pred']).map({True: 'OK', False: 'WRONG'})
print(results.head(20).to_string())

print(f"\nTotal test samples: {len(X_val)}")
print(f"Correct predictions: {(y_val == val_preds).sum()} / {len(X_val)}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
with open("nn_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("\nSaved: nn_model.pkl and scaler.pkl")

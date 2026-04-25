import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

RANDOM_STATE = 55

# Load and preprocess
df = pd.read_csv('heart.csv')
cat_variables = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df = pd.get_dummies(data=df, prefix=cat_variables, columns=cat_variables)

features = [x for x in df.columns if x != 'HeartDisease']
X_train, X_val, y_train, y_val = train_test_split(
    df[features], df['HeartDisease'], train_size=0.8, random_state=RANDOM_STATE
)

print(f"Train samples: {len(X_train)} | Test samples: {len(X_val)}\n")

# --- Decision Tree (tuned: min_samples_split=50, max_depth=3) ---
dt_model = DecisionTreeClassifier(min_samples_split=50, max_depth=3, random_state=RANDOM_STATE)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_val)

# --- Random Forest ---
rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)

# --- XGBoost ---
xgb_model = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_val)

# --- Accuracy Summary ---
print("=" * 50)
print("MODEL ACCURACY ON TEST DATA")
print("=" * 50)
print(f"Decision Tree : {accuracy_score(y_val, dt_preds):.4f}")
print(f"Random Forest : {accuracy_score(y_val, rf_preds):.4f}")
print(f"XGBoost       : {accuracy_score(y_val, xgb_preds):.4f}")

# --- Detailed Report (XGBoost) ---
print("\n" + "=" * 50)
print("XGBOOST CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_val, xgb_preds, target_names=["No Disease", "Disease"]))

# --- Confusion Matrix (XGBoost) ---
print("XGBOOST CONFUSION MATRIX")
print("=" * 50)
cm = confusion_matrix(y_val, xgb_preds)
print(f"                Predicted No  Predicted Yes")
print(f"Actual No       {cm[0][0]:^13} {cm[0][1]:^13}")
print(f"Actual Yes      {cm[1][0]:^13} {cm[1][1]:^13}")

# --- Per-sample predictions on test data ---
print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS ON TEST DATA (first 20 rows)")
print("=" * 50)
results = X_val.copy()
results['Actual'] = y_val.values
results['DT_Pred'] = dt_preds
results['RF_Pred'] = rf_preds
results['XGB_Pred'] = xgb_preds
results['XGB_Correct'] = (results['Actual'] == results['XGB_Pred']).map({True: 'OK', False: 'WRONG'})

display_cols = ['Actual', 'DT_Pred', 'RF_Pred', 'XGB_Pred', 'XGB_Correct']
print(results[display_cols].head(20).to_string())

print(f"\nTotal test samples: {len(X_val)}")
print(f"XGBoost correct predictions: {(y_val.values == xgb_preds).sum()} / {len(X_val)}")

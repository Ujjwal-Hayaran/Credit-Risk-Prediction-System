# train_model.py
# Trains the credit risk model, evaluates it, and saves:
#   - credit_model.pkl, scaler.pkl  (for the Streamlit app)
#   - feature_importance.png, confusion_matrix.png  (for the README)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

from config import EDU_MAP, MARITAL_MAP, RESIDENCE_MAP, PURPOSE_MAP, FEATURE_COLUMNS

# Load the dataset
df = pd.read_csv("dummy_credit_data.csv")

# Encode categorical features using the SAME maps the app uses at inference time
df["education_level"] = df["education_level"].map(EDU_MAP)
df["marital_status"] = df["marital_status"].map(MARITAL_MAP)
df["residence_type"] = df["residence_type"].map(RESIDENCE_MAP)
df["loan_purpose"] = df["loan_purpose"].map(PURPOSE_MAP)

X = df[FEATURE_COLUMNS]
y = df["loan_approved"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC: {auc:.4f}")

# --- Feature importance chart ---
importances = model.feature_importances_
order = np.argsort(importances)[::-1]
sorted_features = [FEATURE_COLUMNS[i] for i in order]
sorted_importances = importances[order]

plt.figure(figsize=(8, 5))
plt.barh(sorted_features[::-1], sorted_importances[::-1], color="#2563eb")
plt.xlabel("Relative Importance")
plt.title("What Drives the Credit Risk Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# --- Confusion matrix chart ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Reject", "Approve"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Save model and scaler
joblib.dump(model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSaved: credit_model.pkl, scaler.pkl, feature_importance.png, confusion_matrix.png")

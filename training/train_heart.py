import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data_path = os.path.join("data", "heart.csv")
df = pd.read_csv(data_path)

# Features & Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"✅ Heart Disease Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/heart_model.pkl")
joblib.dump(scaler, "models/heart_scaler.pkl")

print("🎉 Heart Disease Model saved!")

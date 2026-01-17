import pandas as pd
import os
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ==============================
# 1. LOAD DATA
# ==============================
DATA_PATH = "Data/processed/ipl_next_over_ml.csv"
df = pd.read_csv(DATA_PATH)

# ==============================
# 2. FEATURE SELECTION
# ==============================
FEATURES = [
    "over",
    "runs_in_over",
    "runs_last_3_overs",
    "wickets_last_3_overs",
    "current_run_rate",
    "wickets_remaining",
    "over_phase"
]

TARGET = "runs_next_over"

X = df[FEATURES]
y = df[TARGET]

# ==============================
# 3. TRAIN / TEST SPLIT (TIME SAFE)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ==============================
# 4. MODEL DEFINITION (FINAL)
# ==============================
model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

# ==============================
# 5. TRAIN MODEL
# ==============================
model.fit(X_train, y_train)

# ==============================
# 6. EVALUATION
# ==============================
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("=" * 50)
print(f"FINAL MODEL MAE: {mae:.4f}")
print("=" * 50)

# ==============================
# 7. SAVE MODEL & FEATURES
# ==============================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/next_over_runs_model.pkl")
joblib.dump(FEATURES, "models/features.pkl")

print("Model and features saved successfully.")

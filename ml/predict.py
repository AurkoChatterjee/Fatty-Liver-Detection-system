import torch
import numpy as np
import os
import sys
import pandas as pd

# -------------------------------
# Fix import paths (IMPORTANT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")

sys.path.append(PROJECT_ROOT)

# Now imports will work
from ml.model import NAFLDModel
from ml.preprocess import load_and_preprocess
from llm.explanation_engine import generate_explanation

# -------------------------------
# Load model and scaler
# -------------------------------
model_path = os.path.join(BASE_DIR, "..", "models", "nafld_model.pth")
data_path = os.path.join(BASE_DIR, "..", "data", "final_dataset.csv")

# Load scaler
X, y, scaler = load_and_preprocess(data_path)

# Initialize model
input_size = X.shape[1]
model = NAFLDModel(input_size)

# Load weights safely
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# -------------------------------
# Feature order (VERY IMPORTANT)
# -------------------------------
FEATURE_COLUMNS = [
    "age","male","weight","height","bmi",
    "futime","chol","dbp","fib4","hdl","sbp"
]

# -------------------------------
# Prediction function
# -------------------------------
def predict_risk(patient_data):

    # Convert to DataFrame (fix sklearn warning)
    data_df = pd.DataFrame([patient_data], columns=FEATURE_COLUMNS)

    # Scale
    data_scaled = scaler.transform(data_df)

    # Convert to tensor
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = model(data_tensor).item()

    return prob


# -------------------------------
# Full pipeline: prediction + explanation
# -------------------------------
def predict_with_explanation(patient_data):

    risk = predict_risk(patient_data)

    explanation = generate_explanation(patient_data, risk)

    return risk, explanation


# -------------------------------
# Test example
# -------------------------------
if __name__ == "__main__":

    sample_patient = [
        55,   # age
        1,    # male
        80,   # weight
        170,  # height
        27,   # bmi
        2000, # futime
        60,   # chol
        120,  # dbp
        2.5,  # fib4
        50,   # hdl
        130   # sbp
    ]

    risk, explanation = predict_with_explanation(sample_patient)

    print(f"\n🧠 Predicted Risk: {risk*100:.2f}%")

    if risk > 0.5:
        print("⚠️ High Risk of Fatty Liver")
    else:
        print("✅ Low Risk")

    print("\n🧾 Explanation:\n")
    print(explanation)
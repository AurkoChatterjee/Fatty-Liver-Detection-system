import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):

    df = pd.read_csv(path)

    # Drop useless columns
    df = df.drop(columns=["Unnamed: 0", "id", "case.id", "smoke"], errors="ignore")

    # Fill missing values (mean for numeric)
    df = df.fillna(df.mean(numeric_only=True))

    # Split features and target
    X = df.drop("status", axis=1)
    y = df["status"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler

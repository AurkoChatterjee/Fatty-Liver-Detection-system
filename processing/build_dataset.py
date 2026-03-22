import pandas as pd
import os

# Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths safely
data_path = os.path.join(BASE_DIR, "..", "data")

patients_path = os.path.join(data_path, "nafld1.csv")
labs_path = os.path.join(data_path, "nafld2.csv")

# Load datasets
patients = pd.read_csv(patients_path)
labs = pd.read_csv(labs_path)

# Aggregate lab tests
lab_features = labs.pivot_table(
    index="id",
    columns="test",
    values="value",
    aggfunc="mean"
)

lab_features = lab_features.reset_index()

# Merge
final_df = patients.merge(lab_features, on="id", how="left")

# Save output
output_path = os.path.join(data_path, "final_dataset.csv")
final_df.to_csv(output_path, index=False)

print("✅ Final dataset created!")
print("Shape:", final_df.shape)
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocess import load_and_preprocess
from model import NAFLDModel

# -------------------------------
# 1. Fix Paths (IMPORTANT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "..", "data", "final_dataset.csv")

# -------------------------------
# 2. Load & Preprocess Data
# -------------------------------
X, y, scaler = load_and_preprocess(data_path)

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# -------------------------------
# 4. Model Setup
# -------------------------------
input_size = X_train.shape[1]   # should be 11
model = NAFLDModel(input_size)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 5. Training Loop
# -------------------------------
epochs = 100

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------------------------------
# 6. Evaluation
# -------------------------------
model.eval()

with torch.no_grad():
    preds = model(X_test)
    preds_binary = (preds > 0.5).float()

    accuracy = accuracy_score(y_test, preds_binary)

print("✅ Test Accuracy:", accuracy)

# -------------------------------
# 7. Save Model (FIXED)
# -------------------------------
models_dir = os.path.join(BASE_DIR, "..", "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "nafld_model.pth")

torch.save(model.state_dict(), model_path)

print("💾 Model saved at:", model_path)
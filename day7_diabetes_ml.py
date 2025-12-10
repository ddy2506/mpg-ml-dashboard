import pandas as pd
import numpy as np

# Load dataset (from a trusted public source)
url = "https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/diabetes_nan.csv"
df = pd.read_csv(url)

print("Original shape:", df.shape)
print("Missing values per column:")
print(df.isna().sum())

# Fill missing values with column mean
df = df.fillna(df.mean(numeric_only=True))

print("\nShape after cleaning:", df.shape)

# Select features and target
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# Simple train-test split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Simple Logistic Regression (manual using numpy gradient descent)
weights = np.zeros(X_train.shape[1])
bias = 0
lr = 0.0001

# Train for a few iterations
for _ in range(200):
    linear = np.dot(X_train, weights) + bias
    preds = 1 / (1 + np.exp(-linear))
    dw = np.dot(X_train.T, (preds - y_train)) / len(y_train)
    db = np.mean(preds - y_train)
    weights -= lr * dw
    bias -= lr * db

# Test accuracy
linear_test = np.dot(X_test, weights) + bias
preds_test = 1 / (1 + np.exp(-linear_test))
pred_classes = (preds_test >= 0.5).astype(int)
accuracy = (pred_classes == y_test).mean()

print("\nDiabetes Prediction Model Accuracy:", round(accuracy * 100, 2), "%")

# Example prediction
sample = X_test[0]
risk = 1 / (1 + np.exp(-(np.dot(sample, weights) + bias)))
print("Example risk prediction:", round(risk * 100, 2), "%")

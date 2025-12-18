import pandas as pd
import numpy as np

# 1. Load dataset
url = "https://raw.githubusercontent.com/akmand/datasets/main/california_housing.csv"
print("Loading data from:", url)
df = pd.read_csv(url)

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Keep only numeric columns (drop 'ocean_proximity')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns:", numeric_cols)

target_col = "median_house_value"
feature_cols = [c for c in numeric_cols if c != target_col]

# Fill any numeric NaNs with column means (safety)
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

X = df[feature_cols].values
y = df[target_col].values

print("\nNumber of numeric features:", X.shape[1])

# 2. Train / test split
n = len(X)
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Feature scaling (helps gradient descent)
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_std[X_std == 0] = 1.0  # avoid division by zero

X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# 4. Gradient descent linear regression
n_features = X_train_scaled.shape[1]
w = np.zeros(n_features)
b = 0.0
lr = 0.01
epochs = 1000

for epoch in range(epochs):
    y_pred_train = X_train_scaled @ w + b
    error = y_pred_train - y_train

    dw = (X_train_scaled.T @ error) / len(X_train_scaled)
    db = error.mean()

    w -= lr * dw
    b -= lr * db

    if epoch % 200 == 0:
        mse = np.mean(error ** 2)
        rmse = np.sqrt(mse)
        print(f"Epoch {epoch:4d} | RMSE: {rmse:,.1f}")

print("\nTraining finished.")
print("Weights (first 5):", [round(val, 5) for val in w[:5]])
print("Bias:", round(b, 3))

# 5. Evaluate on test set
y_pred_test = X_test_scaled @ w + b
mse_test = np.mean((y_pred_test - y_test) ** 2)
rmse_test = np.sqrt(mse_test)

print("\nTest RMSE:", round(rmse_test, 1))

# 6. Show a few predictions
print("\nExample predictions on test data (first 5):")
for i in range(5):
    print(f"True: {y_test[i]:,.1f} | Predicted: {y_pred_test[i]:,.1f}")

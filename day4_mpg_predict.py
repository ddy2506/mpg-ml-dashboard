import pandas as pd
import numpy as np

# Load dataset
url = "https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/mpg.csv"
df = pd.read_csv(url)

# Show columns once (for learning/debug)
print("Columns in dataset:")
print(df.columns)

# Keep only rows where both mpg and horsepower are not missing
df = df[["mpg", "horsepower"]].dropna()

# X = horsepower, y = mpg
x = df["horsepower"].values
y = df["mpg"].values

# Simple linear regression (manual with numpy)
x_mean = x.mean()
y_mean = y.mean()

numerator = ((x - x_mean) * (y - y_mean)).sum()
denominator = ((x - x_mean) ** 2).sum()
a = numerator / denominator
b = y_mean - a * x_mean

print("\nSimple MPG prediction model:")
print("mpg ≈ a * horsepower + b")
print(f"a (slope): {a:.4f}")
print(f"b (intercept): {b:.4f}")

# Try a few predictions
test_hp = [80, 120, 160]

print("\nExample predictions:")
for hp in test_hp:
    mpg_pred = a * hp + b
    print(f"Horsepower = {hp:3d} → predicted MPG ≈ {mpg_pred:.2f}")

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create output folder for charts
output_dir = "/Users/ddy2025/Python3/charts"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
url = "https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/mpg.csv"
df = pd.read_csv(url)

# Keep only mpg and horsepower, remove missing
df = df[["mpg", "horsepower"]].dropna()

x = df["horsepower"].values
y = df["mpg"].values

# --- Train simple linear regression (same as before) ---
x_mean = x.mean()
y_mean = y.mean()

a = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
b = y_mean - a * x_mean

print("Model:")
print(f"mpg ≈ {a:.4f} * horsepower + {b:.4f}")

# --- Create prediction line ---
x_line = np.linspace(x.min(), x.max(), 100)
y_line = a * x_line + b

# --- Plot data + prediction line ---
fig = plt.figure()
plt.scatter(x, y, alpha=0.5, label="Real data")
plt.plot(x_line, y_line, label="Prediction line")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("MPG vs Horsepower with Prediction Line")
plt.grid(True)
plt.legend()

# Save as PNG
plot_path = os.path.join(output_dir, "mpg_ml_prediction_line.png")
fig.savefig(plot_path, bbox_inches="tight")
plt.close(fig)

print("Saved ML plot to:", plot_path)

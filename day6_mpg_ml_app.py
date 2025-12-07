import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Car MPG Prediction Demo (Horsepower → MPG)")

# Load dataset
url = "https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/mpg.csv"
df = pd.read_csv(url)

st.write("### Data preview")
st.dataframe(df[["mpg", "horsepower", "weight", "origin"]].head())

# Clean data: keep only rows with mpg & horsepower
df_clean = df[["mpg", "horsepower"]].dropna()

x = df_clean["horsepower"].values
y = df_clean["mpg"].values

# --- Train simple linear regression (same as before) ---
x_mean = x.mean()
y_mean = y.mean()

a = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
b = y_mean - a * x_mean

st.write("### Learned Model")
st.write(f"mpg ≈ {a:.4f} × horsepower + {b:.4f}")

# --- User input for prediction ---
st.write("### Try a Prediction")

hp_input = st.number_input(
    "Enter horsepower:",
    min_value=float(x.min()),
    max_value=float(x.max()),
    value=float(100),
    step=5.0,
)

mpg_pred = a * hp_input + b
st.write(f"**Predicted MPG for {hp_input:.0f} HP:** {mpg_pred:.2f}")

# --- Plot data + prediction line ---
st.write("### MPG vs Horsepower with Prediction Line")

x_line = np.linspace(x.min(), x.max(), 100)
y_line = a * x_line + b

fig = plt.figure()
plt.scatter(x, y, alpha=0.5, label="Real data")
plt.plot(x_line, y_line, label="Prediction line")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("MPG vs Horsepower with Prediction Line")
plt.grid(True)
plt.legend()

st.pyplot(fig)

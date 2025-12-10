import pandas as pd
import numpy as np
import streamlit as st

st.title("Diabetes Risk Prediction Demo")

# Load dataset locally
csv_path = "/Users/ddy2025/Python3/diabetes_local.csv"

# If you don't have a local copy yet, load from web once and save
if not st.session_state.get("data_loaded", False):
    url = "https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/diabetes_nan.csv"
    df = pd.read_csv(url)
    df.to_csv(csv_path, index=False)
    st.session_state["data_loaded"] = True
else:
    df = pd.read_csv(csv_path)

st.write("### Data Preview")
st.dataframe(df.head())

# Clean missing values
df = df.fillna(df.mean(numeric_only=True))

# Features & target
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# Train model (same as before)
weights = np.zeros(X.shape[1])
bias = 0
lr = 0.0001

for _ in range(200):
    linear = np.dot(X, weights) + bias
    preds = 1 / (1 + np.exp(-linear))
    dw = np.dot(X.T, (preds - y)) / len(y)
    db = np.mean(preds - y)
    weights -= lr * dw
    bias -= lr * db

# User input
st.write("## Enter Patient Data")

inputs = []
for col in df.drop("Outcome", axis=1).columns:
    val = st.number_input(col, float(df[col].min()), float(
        df[col].max()), float(df[col].mean()))
    inputs.append(val)

inputs = np.array(inputs)

# Prediction
risk = 1 / (1 + np.exp(-(np.dot(inputs, weights) + bias)))

st.write("### Predicted Diabetes Risk")
st.write(f"**Risk Probability: {risk*100:.2f}%**")

if risk >= 0.5:
    st.error("🔴 High Risk of Diabetes (demo model)")
else:
    st.success("🟢 Low Risk of Diabetes (demo model)")

st.caption("⚠️ This is an educational demo model only, not medical advice.")

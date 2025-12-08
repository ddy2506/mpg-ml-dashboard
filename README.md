# Car MPG Machine Learning Dashboard

This project demonstrates a complete mini data science and machine learning workflow using Python.  
It analyzes car fuel efficiency (MPG) and builds a simple model to predict MPG from engine horsepower.

## 🔍 What this project does

- Loads a real-world car MPG dataset (stored locally as `mpg_local.csv`)
- Cleans the data (removing missing values in MPG and horsepower)
- Explores the data with:
  - Average MPG by origin (USA / Europe / Japan)
  - MPG distribution histogram
- Trains a simple linear regression model:
  - Input (feature): `horsepower`
  - Output (target): `mpg`
  - Model: `mpg ≈ a × horsepower + b`
- Visualizes:
  - Scatter plot of MPG vs horsepower
  - Prediction line from the model
- Provides an interactive **Streamlit web app** for:
  - Viewing the data
  - Seeing the model equation
  - Entering a horsepower value and getting a predicted MPG

## 🛠 Technologies Used

- Python
- pandas
- numpy
- matplotlib
- Streamlit
- Git & GitHub

## 📊 Generated Charts

Saved in the `charts/` folder:

- `avg_mpg_by_origin.png` – Average MPG by origin
- `mpg_distribution.png` – Histogram of MPG values
- `mpg_ml_prediction_line.png` – MPG vs horsepower with prediction line

These plots can be used in reports, CVs, or presentations.

## ▶️ How to Run the Streamlit App

1. Clone this repository:

   ```bash
   git clone https://github.com/ddy2506/mpg-ml-dashboard.git
   cd mpg-ml-dashboard

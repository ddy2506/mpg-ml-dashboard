import pandas as pd

file_path = "/Users/ddy2025/Downloads/my_data.csv"

try:
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
    print(df.head())
except Exception as e:
    print("Error reading file:", e)

import pandas as pd

# Load real dataset from the web (no need to download)
url = "https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/mpg.csv"
df = pd.read_csv(url)

print(df.head())

# Show column names

print("\nColumns in dataset:")
print(df.columns)

# Show basic info
print("\nDataset info:")
print(df.info())

# Show basic statistics
print("\nBasic statistics:")
print(df.describe())

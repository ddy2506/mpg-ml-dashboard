import pandas as pd

# Real dataset from the web (no need to download)
url = "https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/mpg.csv"
df = pd.read_csv(url)

print("✅ File loaded!")
print("Shape (rows, columns):", df.shape)
print("\nColumn names:\n", df.columns)
print("\nFirst 5 rows:")
print(df.head())

print("\nAverage MPG by origin:")
print(df.groupby("origin")["mpg"].mean())

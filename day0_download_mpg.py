import pandas as pd

url = "https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/mpg.csv"

print("Downloading dataset...")
df = pd.read_csv(url)
save_path = "/Users/ddy2025/Python3/mpg_local.csv"
df.to_csv(save_path, index=False)

print("Saved local copy to:", save_path)
print("Shape:", df.shape)

import os
import pandas as pd
import matplotlib.pyplot as plt

# Output folder
output_dir = "/Users/ddy2025/Python3/charts"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
url = "https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/mpg.csv"
df = pd.read_csv(url)

# ---------- BAR CHART ----------
avg_mpg = df.groupby("origin")["mpg"].mean()

fig1 = plt.figure()
avg_mpg.plot(kind="bar")
plt.title("Average MPG by Origin")
plt.xlabel("Origin")
plt.ylabel("MPG")
plt.grid(True)

bar_path = os.path.join(output_dir, "avg_mpg_by_origin.png")
fig1.savefig(bar_path, bbox_inches="tight")
plt.close(fig1)   # ✅ VERY IMPORTANT
print("Saved:", bar_path)

# ---------- HISTOGRAM ----------
fig2 = plt.figure()
plt.hist(df["mpg"], bins=10)
plt.title("MPG Distribution")
plt.xlabel("MPG")
plt.ylabel("Frequency")
plt.grid(True)

hist_path = os.path.join(output_dir, "mpg_distribution.png")
fig2.savefig(hist_path, bbox_inches="tight")
plt.close(fig2)   # ✅ VERY IMPORTANT
print("Saved:", hist_path)

import pandas as pd
import matplotlib.pyplot as plt

# Load real dataset
url = "https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/mpg.csv"
df = pd.read_csv(url)

# ---------- BAR CHART: Average MPG by Origin ----------
avg_mpg = df.groupby("origin")["mpg"].mean()

plt.figure()
avg_mpg.plot(kind="bar")
plt.title("Average MPG by Origin")
plt.xlabel("Origin")
plt.ylabel("MPG")
plt.grid(True)
plt.show()

# ---------- HISTOGRAM: MPG Distribution ----------
plt.figure()
plt.hist(df["mpg"], bins=10)
plt.title("MPG Distribution")
plt.xlabel("MPG")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

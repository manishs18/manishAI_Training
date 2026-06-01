"""
Simple flat script version of the capstone solution.

No loops or if/else statements — linear and easy to read.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

df = sns.load_dataset("taxis")
print("Section 1 — Load & Inspect")
print("Source: seaborn:taxis")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include="all"))
print("\nThis dataset contains taxi trip rows; each row represents one trip event.\n")

print("Section 2 — Clean & Prepare")
print("Shape before:", df.shape)
print(df.isna().sum())

df = df.dropna(subset=["fare", "tip"])
num_cols = df.select_dtypes(include=[np.number]).columns
medians = df[num_cols].median()
df[num_cols] = df[num_cols].fillna(medians)

df["tip_pct"] = np.where(df["fare"] > 0, df["tip"] / df["fare"] * 100, 0)
print("Shape after:", df.shape)
print("Dropped rows missing fare or tip and filled numeric NA with medians. Added tip_pct.")

print("Section 3 — Explore with Statistics")
print("Fare percentiles:", np.percentile(df["fare"], [25, 50, 75]))
corr = np.corrcoef(df["fare"], df["tip"])[0, 1]
print("Correlation fare vs tip:", corr)
print(df.groupby("payment")["fare"].agg(["mean", "median"]))
print("Summary stats:")
print("Median fare:", np.median(df["fare"]))
print("Median tip %:", np.median(df["tip_pct"]))

print("Section 4 — Visualise")
plt.figure(figsize=(8, 5))
plt.hist(df["fare"], bins=40, color="#4c72b0")
plt.title("Distribution of Fare")
plt.xlabel("Fare")
plt.ylabel("Count")
path1 = os.path.join(OUT, "simple_chart_fare_hist.png")
plt.tight_layout(); plt.savefig(path1); plt.close()
print("Saved:", path1)

sample = df.sample(n=2000, random_state=1)
plt.figure(figsize=(8, 5))
sns.regplot(x="fare", y="tip", data=sample, scatter_kws={"s": 10})
plt.title("Tip vs Fare")
plt.xlabel("Fare")
plt.ylabel("Tip")
med = np.median(df["fare"])
plt.axvline(med, color="red", linestyle="--")
plt.annotate(f"Median fare: {med:.2f}", xy=(med, plt.gca().get_ylim()[1] * 0.8), color="red")
path2 = os.path.join(OUT, "simple_chart_tip_vs_fare.png")
plt.tight_layout(); plt.savefig(path2); plt.close()
print("Saved:", path2)

plt.figure(figsize=(8, 5))
sns.boxplot(x="payment", y="tip_pct", data=df)
plt.title("Tip % by Payment Method")
path3 = os.path.join(OUT, "simple_chart_tippct_by_payment.png")
plt.tight_layout(); plt.savefig(path3); plt.close()
print("Saved:", path3)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.countplot(x="passengers", data=df, ax=axes[0])
axes[0].set_title("Trips by Passenger Count")
hr = df.groupby("passengers")["fare"].mean().reset_index()
sns.lineplot(x="passengers", y="fare", data=hr, marker="o", ax=axes[1])
axes[1].set_title("Average Fare by Passenger Count")
fig.suptitle("Trips and Fare by Passenger Count")
path4 = os.path.join(OUT, "multi_panel_passengers.png")
plt.tight_layout(); fig.savefig(path4); plt.close(fig)
print("Saved:", path4)

print("Section 5 — Key Insights")
print("1. Card payments show higher median tip % than cash (see simple_chart_tippct_by_payment.png).")
print("2. Tip and fare are positively correlated (see simple_chart_tip_vs_fare.png).")
print("3. Fare distribution is right-skewed (see simple_chart_fare_hist.png).")

print("Section 6 — Narrative Conclusion")
print("We loaded taxi trip data, cleaned missing values, engineered tip_pct, calculated statistics, and created charts. The results show fare-tip correlation and a higher tip percentage for card payments. Next, a decision-maker could test payment incentives and monitor tip behaviour over time.")

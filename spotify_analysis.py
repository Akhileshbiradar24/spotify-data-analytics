"""
CES 2103 - Data Analytics | IA 3 Mini Project
Spotify Top 10,000 tracks (1960-now): Preprocessing, EDA, and CDA
Tests used: Pearson correlation, Chi-Square of Independence, Simple & Multiple OLS Regression
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

ALPHA = 0.05
FEATURES = [
    "Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Valence", "Tempo"
]

RAW_FILE = "top_10000_1960-now.csv"
CLEAN_FILE = "spotify_cleaned.csv"

if os.path.exists(RAW_FILE):
    print(f"--> Found raw data '{RAW_FILE}'. Running full cleaning pipeline...")
    d = pd.read_csv(RAW_FILE)
    print("Raw dataset shape:", d.shape)

    n = len(d)
    d = d.drop_duplicates("Track URI")
    print("Duplicates removed:", n - len(d))

    n = len(d)
    d = d.dropna(subset=FEATURES + ["Track Name"])
    print("Rows with missing values removed:", n - len(d))

    d["year"] = pd.to_datetime(d["Album Release Date"], errors="coerce").dt.year
    n = len(d)
    d = d.dropna(subset=["year"])
    d["year"] = d["year"].astype(int)
    print("Rows without valid year removed:", n - len(d))

    n = len(d)
    d = d[d["Popularity"] > 0]
    print("Popularity = 0 rows removed:", n - len(d))

    d["Explicit"] = d["Explicit"].astype(str).str.lower().eq("true").astype(int)
    d["decade"] = (d["year"] // 10 * 10).clip(lower=1960)
    median_pop = d["Popularity"].median()
    d["popcat"] = np.where(d["Popularity"] >= median_pop, "High", "Low")
    print("Final clean shape:", d.shape, "| Popularity median threshold:", median_pop)

    keep = [
        "Track Name", "Artist Name(s)", "Album Name", "Album Release Date", "year", "decade",
        "Explicit", "Popularity", "popcat", "Track Duration (ms)", "Key", "Mode",
        "Time Signature"
    ] + FEATURES
    d[keep].to_csv(CLEAN_FILE, index=False)
    print(f"Cleaned dataset saved as '{CLEAN_FILE}'.\n")

elif os.path.exists(CLEAN_FILE):
    print(f"--> Raw file not found. Loading preprocessed '{CLEAN_FILE}' directly...")
    d = pd.read_csv(CLEAN_FILE)
    print("Loaded dataset shape:", d.shape, "\n")
else:
    raise FileNotFoundError(
        f"Neither '{RAW_FILE}' nor '{CLEAN_FILE}' was found in the current folder. "
        "Please ensure at least one of these CSV files is placed in the script directory."
    )

print("================== DESCRIPTIVE STATISTICS ==================")
print(d[FEATURES + ["Popularity", "year"]].describe().T.round(3))
print("\nOverall Explicit %:", round(d["Explicit"].mean() * 100, 1))
print("\nMean Popularity by Decade:\n", d.groupby("decade")["Popularity"].mean().round(1))
print("\n% Explicit Tracks by Decade:\n", (d.groupby("decade")["Explicit"].mean() * 100).round(1))

plt.rcParams.update({"font.size": 9})

print("\nDisplaying Figure 1 (Distributions)... Close the plot window to continue.")
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for a, col, c in zip(ax, ["Popularity", "year", "Danceability"], ["#2a6f97", "#61a5c2", "#89c2d9"]):
    a.hist(d[col], bins=30, color=c)
    a.set_title(col)
plt.tight_layout()
plt.savefig("fig1_distributions.png", dpi=160)
plt.show()

print("Displaying Figure 2 (Correlation & Decade Trend)... Close the plot window to continue.")
cols = FEATURES + ["Popularity", "year"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
im = ax[0].imshow(d[cols].corr(), cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_xticks(range(len(cols)))
ax[0].set_yticks(range(len(cols)))
ax[0].set_xticklabels(cols, rotation=90)
ax[0].set_yticklabels(cols)
ax[0].set_title("Correlation Matrix")
plt.colorbar(im, ax=ax[0], fraction=0.046)

m = d.groupby("decade")["Popularity"].mean()
ax[1].bar(m.index.astype(str), m.values, color="#2a6f97")
ax[1].set_title("Mean Popularity by Decade")
ax[1].set_ylim(40, 70)
plt.tight_layout()
plt.savefig("fig2_correlation_decade.png", dpi=160)
plt.show()

print("Displaying Figure 3 (Scatter Fit & Explicit Trend)... Close the plot window to continue.")
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].scatter(d["Energy"], d["Loudness"], s=3, alpha=0.3, color="#2a6f97")
b = np.polyfit(d["Energy"], d["Loudness"], 1)
xs = np.linspace(0, 1, 10)
ax[0].plot(xs, np.polyval(b, xs), "r")
ax[0].set_xlabel("Energy")
ax[0].set_ylabel("Loudness (dB)")
ax[0].set_title("Energy vs Loudness")

e = d.groupby("decade")["Explicit"].mean() * 100
ax[1].bar(e.index.astype(str), e.values, color="#61a5c2")
ax[1].set_title("% Explicit Tracks by Decade")
plt.tight_layout()
plt.savefig("fig3_energy_loudness_explicit.png", dpi=160)
plt.show()

def decision(p): 
    return "Reject H0" if p < ALPHA else "Fail to reject H0"

print("\n================== CDA HYPOTHESIS TESTING ==================")

print("\n--- TEST 1: Pearson Correlation (H0: rho = 0) ---")
pairs = [
    ("Energy", "Loudness"), ("Danceability", "Valence"), ("Acousticness", "Energy"),
    ("Danceability", "Popularity"), ("year", "Popularity")
]
for a, b_ in pairs:
    r, p = st.pearsonr(d[a], d[b_])
    print(f"{a:13s} vs {b_:11s}: r = {r:+.3f}, p = {p:.3g} -> {decision(p)}")

print("\n--- TEST 2: Chi-Square Test (Explicit vs Popularity Category) ---")
ct = pd.crosstab(d["Explicit"], d["popcat"])
print(ct)
chi2, p, dof, _ = st.chi2_contingency(ct)
print(f"chi2 = {chi2:.2f}, df = {dof}, p = {p:.3g} -> {decision(p)}")
print("% High popularity:", (ct["High"] / ct.sum(axis=1) * 100).round(1).to_dict())

print("\n--- TEST 3: Chi-Square Test (Decade vs Explicit) ---")
ct2 = pd.crosstab(d["decade"], d["Explicit"])
print(ct2)
chi2, p, dof, _ = st.chi2_contingency(ct2)
v = np.sqrt(chi2 / (ct2.values.sum() * (min(ct2.shape) - 1)))
print(f"chi2 = {chi2:.2f}, df = {dof}, p = {p:.3g}, Cramer's V = {v:.3f} -> {decision(p)}")

print("\n--- TEST 4: Simple Linear Regression (Loudness ~ Energy) ---")
lr = st.linregress(d["Energy"], d["Loudness"])
print(f"Loudness = {lr.intercept:.2f} + {lr.slope:.2f} * Energy | R2 = {lr.rvalue**2:.3f} | p = {lr.pvalue:.3g}")

print("\n--- TEST 5: Multiple Linear Regression (Popularity ~ Features + Explicit + year) ---")
X = sm.add_constant(d[FEATURES + ["Explicit", "year"]])
model = sm.OLS(d["Popularity"], X).fit()
print(model.summary())
print(f"\nFinal Summary: R2 = {model.rsquared:.4f}, adj R2 = {model.rsquared_adj:.4f}, F = {model.fvalue:.2f}, p = {model.f_pvalue:.3g}")
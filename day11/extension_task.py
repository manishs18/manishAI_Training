# time_series_eda_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import mode

# ==========================================
# 1. LOAD DATA
# ==========================================
url = "https://raw.githubusercontent.com/rajatg1989/sakshi/main/11%2006%202026/sakshi%20ppg%202%200260611T074737%20len148s.csv"

df = pd.read_csv(url)
print("Dataset Loaded")
print(df.head())

# Assuming the first column is timestamp and second column is PPG values
df.columns = ['timestamp', 'ppg']
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# ==========================================
# 2. CHECK TIME CONTINUITY AND MISSING TIMESTAMPS
# ==========================================
# Assuming regular sampling (infer frequency)
df = df.asfreq(pd.infer_freq(df.index))
missing_ts = df['ppg'].isna().sum()
print(f"Missing timestamps / values: {missing_ts}")

# ==========================================
# 3. HANDLE MISSING VALUES
# ==========================================
# Fill missing values by interpolation (linear)
df['ppg'] = df['ppg'].interpolate(method='linear')
print("Missing values after interpolation:", df['ppg'].isna().sum())

# ==========================================
# 4. BASIC STATS
# ==========================================
print("\n=== BASIC STATISTICS ===")
print("Mean:", df['ppg'].mean())
print("Std:", df['ppg'].std())
print("Mode:", mode(df['ppg'])[0][0])

# ==========================================
# 5. TIME SERIES PLOT
# ==========================================
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['ppg'], label='PPG')
plt.title('PPG Time Series')
plt.xlabel('Time')
plt.ylabel('PPG')
plt.legend()
plt.show()

# ==========================================
# 6. DECOMPOSITION
# ==========================================
# Estimate seasonal period: e.g., 50 for ~1Hz PPG with 50Hz sampling
period = 50
decomposition = seasonal_decompose(df['ppg'], model='additive', period=period)

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(decomposition.observed, label='Observed')
plt.legend()
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.legend()
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal', color='green')
plt.legend()
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual', color='red')
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 7. OUTLIER DETECTION (Z-score method)
# ==========================================
z_scores = (df['ppg'] - df['ppg'].mean()) / df['ppg'].std()
outliers = df[np.abs(z_scores) > 3]
print(f"Number of outliers detected: {len(outliers)}")

# ==========================================
# 8. BOX PLOTS (weekly & monthly)
# ==========================================
df['month'] = df.index.month
df['week'] = df.index.isocalendar().week

plt.figure(figsize=(12, 4))
sns.boxplot(x='month', y='ppg', data=df)
plt.title("Monthly PPG Distribution")
plt.show()

plt.figure(figsize=(12, 4))
sns.boxplot(x='week', y='ppg', data=df)
plt.title("Weekly PPG Distribution")
plt.show()

# ==========================================
# 9. STATIONARITY CHECK
# ==========================================
result = adfuller(df['ppg'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] < 0.05:
    print("Time series is stationary")
else:
    print("Time series is non-stationary")

# ==========================================
# 10. ACF AND PACF
# ==========================================
lag_acf = acf(df['ppg'], nlags=50)
lag_pacf = pacf(df['ppg'], nlags=50, method='ols')

plt.figure(figsize=(12, 4))
plt.stem(range(len(lag_acf)), lag_acf, use_line_collection=True)
plt.title('Autocorrelation (ACF)')
plt.show()

plt.figure(figsize=(12, 4))
plt.stem(range(len(lag_pacf)), lag_pacf, use_line_collection=True)
plt.title('Partial Autocorrelation (PACF)')
plt.show()

# ==========================================
# 11. FEATURE ENGINEERING COLUMNS
# ==========================================
# Add rolling statistics as features
df['ppg_roll_mean_5'] = df['ppg'].rolling(window=5).mean()
df['ppg_roll_std_5'] = df['ppg'].rolling(window=5).std()

df['ppg_roll_mean_15'] = df['ppg'].rolling(window=15).mean()
df['ppg_roll_std_15'] = df['ppg'].rolling(window=15).std()

print("Feature Engineering columns added:")
print(df.columns)

# ==========================================
# 12. MISSING VALUE STRATEGY
# ==========================================
missing_percentage = df.isna().sum() / len(df) * 100
print("\nMissing value %age:")
print(missing_percentage)
print("Strategy: Linear interpolation, then forward/backward fill if needed")

# ==========================================
# PIPELINE COMPLETE
# ==========================================
print("\nEDA pipeline completed successfully. Artifacts generated:")
print("1. Time series plot")
print("2. Decomposition plot (trend, seasonality, residuals)")
print("3. Outlier detection")
print("4. Boxplots (weekly/monthly)")
print("5. Stationarity test results")
print("6. ACF / PACF plots")
print("7. Feature engineering columns added")
print("8. Missing value percentage and fill strategy")
import pandas as pd
import os

from config import *

from data_loader import load_data
from eda import summary_stats, missing_report
from feature_engineering import create_features

from time_series_analysis import (
    plot_series,
    decompose_series,
    stationarity_test,
    plot_acf_pacf,
    boxplots
)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(DECOMP_DIR, exist_ok=True)

df = load_data(DATA_PATH)

df[TIME_COLUMN] = (
    pd.to_datetime(df[TIME_COLUMN])
)

df = df.sort_values(TIME_COLUMN)

df.set_index(TIME_COLUMN, inplace=True)

# Missing Report
missing = missing_report(df)

missing.to_csv(
    os.path.join(
        REPORT_DIR,
        "missing_values_report.csv"
    ),
    index=False
)

# Fill Missing Values
df = df.ffill().bfill()

# Statistics
stats = summary_stats(df, TARGET_COLUMN)

stats.to_csv(
    os.path.join(
        REPORT_DIR,
        "summary_statistics.csv"
    ),
    index=False
)

# Feature Engineering
feature_df = create_features(
    df,
    TARGET_COLUMN
)

feature_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "feature_engineering_report.csv"
    )
)

# Time Series Plot
plot_series(
    df,
    TARGET_COLUMN,
    os.path.join(
        PLOT_DIR,
        "time_series.png"
    )
)

# Boxplots
boxplots(
    df,
    TARGET_COLUMN,
    PLOT_DIR
)

# Decomposition
decompose_series(
    df,
    TARGET_COLUMN,
    period=30,
    output_dir=DECOMP_DIR
)

# Stationarity
report = stationarity_test(
    df[TARGET_COLUMN]
)

with open(
    os.path.join(
        REPORT_DIR,
        "stationarity_report.txt"
    ),
    "w"
) as f:
    f.write(report)

# ACF PACF
plot_acf_pacf(
    df[TARGET_COLUMN],
    PLOT_DIR
)

print("Pipeline Completed Successfully")
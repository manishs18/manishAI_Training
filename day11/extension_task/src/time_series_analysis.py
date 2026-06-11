import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

def plot_series(df, target_col, save_path):

    plt.figure(figsize=(14,5))
    plt.plot(df.index, df[target_col])
    plt.title("Time Series Plot")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def decompose_series(df,
                     target_col,
                     period,
                     output_dir):

    result = seasonal_decompose(
        df[target_col],
        model='additive',
        period=period
    )

    result.trend.to_csv(
        os.path.join(output_dir, "trend.csv")
    )

    result.seasonal.to_csv(
        os.path.join(output_dir, "seasonal.csv")
    )

    result.resid.to_csv(
        os.path.join(output_dir, "residual.csv")
    )

    fig = result.plot()
    fig.set_size_inches(12,8)

    plt.savefig(
        os.path.join(
            output_dir,
            "../plots/decomposition.png"
        )
    )

    plt.close()

def stationarity_test(series):

    adf = adfuller(series.dropna())

    report = f"""
ADF Statistic: {adf[0]}
P-value: {adf[1]}
"""

    return report

def plot_acf_pacf(series, plot_dir):

    fig = plot_acf(series.dropna(), lags=50)
    plt.savefig(
        os.path.join(plot_dir, "acf.png")
    )
    plt.close()

    fig = plot_pacf(series.dropna(), lags=50)
    plt.savefig(
        os.path.join(plot_dir, "pacf.png")
    )
    plt.close()

def boxplots(df, target_col, plot_dir):

    plt.figure(figsize=(10,5))
    sns.boxplot(
        x=df.index.month,
        y=df[target_col]
    )

    plt.title("Monthly Boxplot")

    plt.savefig(
        os.path.join(
            plot_dir,
            "monthly_boxplot.png"
        )
    )

    plt.close()

    plt.figure(figsize=(10,5))
    sns.boxplot(
        x=df.index.isocalendar().week,
        y=df[target_col]
    )

    plt.title("Weekly Boxplot")

    plt.savefig(
        os.path.join(
            plot_dir,
            "weekly_boxplot.png"
        )
    )

    plt.close()
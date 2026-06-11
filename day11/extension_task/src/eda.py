import pandas as pd
import numpy as np
from scipy.stats import mode

def summary_stats(df, target_col):

    result = {
        "mean": df[target_col].mean(),
        "std": df[target_col].std(),
        "median": df[target_col].median(),
        "min": df[target_col].min(),
        "max": df[target_col].max(),
        "mode": df[target_col].mode().iloc[0]
    }

    return pd.DataFrame([result])

def missing_report(df):

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isnull().sum(),
        "missing_percent":
            round(df.isnull().mean()*100,2)
    })

    return report
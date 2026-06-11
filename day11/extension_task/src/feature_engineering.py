def create_features(df, target_col):

    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["weekday"] = df.index.dayofweek

    df[f"{target_col}_lag1"] = df[target_col].shift(1)
    df[f"{target_col}_lag2"] = df[target_col].shift(2)

    df[f"{target_col}_rolling_mean"] = (
        df[target_col]
        .rolling(window=10)
        .mean()
    )

    df[f"{target_col}_rolling_std"] = (
        df[target_col]
        .rolling(window=10)
        .std()
    )

    df[f"{target_col}_diff"] = (
        df[target_col]
        .diff()
    )

    return df
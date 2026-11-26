import pandas as pd


def azureml_main(dataframe1=None, dataframe2=None):
    df = dataframe1.copy()

    # Make sure the date is a real datetime and the rows are chronological
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    dt = df["Datetime"]
    df["Hour"] = dt.dt.hour
    df["DayOfWeek"] = dt.dt.dayofweek      # 0 = Monday
    df["Month"] = dt.dt.month
    df["Quarter"] = dt.dt.quarter
    df["Year"] = dt.dt.year
    df["DayOfYear"] = dt.dt.dayofyear
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    return df,
import numpy as np
import pandas as pd


def skew_calc(df):
    """
    Diagnoses skewness for every numeric column in a DataFrame and recommends a transformation based on the column's skewness and
    minimum value. Binary, encoded, and ID columns are excluded, since skewness isn't a meaningful for them.
    It returns a DataFrame with the following columns:
    Feature, Skewness, Degree, Direction, Recommended Transformation
    """

    results = []

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        # Skip binary and ID columns
        if df[col].nunique() <= 2:
            continue

        if "id" in col.lower():
            continue

        skew = df[col].dropna().skew()

        # Degree
        if abs(skew) < 0.5:
            degree = "Approximately Symmetric"
            recommendation = "None needed"

        elif abs(skew) < 1:
            degree = "Moderately Skewed"

            if skew > 0:
                recommendation = "log(x+1) or Yeo-Johnson"
            else:
                recommendation = "Box-Cox or Yeo-Johnson"

        else:
            degree = "Highly Skewed"
            recommendation = "Box-Cox or Yeo-Johnson"

        # Direction
        if skew >= 0:
            direction = "Positive"
        else:
            direction = "Negative"

        results.append({
            "Feature": col,
            "Skewness": skew,
            "Degree": degree,
            "Direction": direction,
            "Recommended Transformation": recommendation
        })

    return pd.DataFrame(results)

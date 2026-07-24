import numpy as np
import pandas as pd

def skew_calc(df):
    """
    Diagnoses skewness for every numeric column in a DataFrame and recommends a transformation based on the column's skewness and
    minimum value. Binary, encoded, and ID columns are excluded, since skewness isn't a meaningful for them.
    It returns a DataFrame with the following columns:
    Feature, Skewness, Degree, Direction, Recommended Transformation
    """
    records = []

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        series = df[col].dropna()

        # Skip binary / near-constant columns -- skewness isn't meaningful for them
        if series.nunique() <= 2:
            continue

        # Skip likely ID / encoded columns based on naming convention
        lowered = col.lower()
        if 'id' in lowered or 'tier' in lowered or 'encoded' in lowered or 'binary' in lowered:
            continue

        skew_val = series.skew()

        # Degree/direction classification (Table 1)
        if skew_val < -1:
            degree, direction = 'Highly Skewed', 'Left'
        elif -1 <= skew_val < -0.5:
            degree, direction = 'Moderately Skewed', 'Left'
        elif -0.5 <= skew_val <= 0.5:
            degree, direction = 'Normal (Symmetrical)', 'None'
        elif 0.5 < skew_val <= 1:
            degree, direction = 'Moderately Skewed', 'Right'
        else:
            degree, direction = 'Highly Skewed', 'Right'

        # Recommended transformation (Table 2)
        if degree == 'Normal (Symmetrical)':
            recommendation = 'None'
        elif series.min() < 0:
            recommendation = 'Yeo-Johnson'
        elif series.min() == 0:
            recommendation = 'Log Plus One'
        else:
            recommendation = 'Box-Cox'

        records.append({
            'Feature': col,
            'Skewness': round(skew_val, 4),
            'Degree': degree,
            'Direction': direction,
            'Recommended Transformation': recommendation
        })

    return pd.DataFrame(records)

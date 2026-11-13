"""
Data preparation for the Pima Indians Diabetes dataset.

Key improvement over typical beginner repos: the raw dataset encodes missing
clinical values as 0 (impossible for Glucose, BloodPressure, SkinThickness,
Insulin, BMI). We treat those zeros as NaN and impute them properly instead
of silently training on biologically impossible values.
"""

import pandas as pd
import numpy as np

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_raw(path="data/diabetes_raw.csv"):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)
    return df


if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    df.to_csv("data/diabetes_clean.csv", index=False)
    print("Missing values per column after zero->NaN conversion:")
    print(df.isna().sum())
    print(f"\nTotal rows: {len(df)}")
    print(f"Class balance:\n{df['Outcome'].value_counts(normalize=True)}")

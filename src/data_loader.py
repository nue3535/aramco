import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    Args:
        path (str): Path to the CSV file
    Returns:
        pd.DataFrame: Loaded data
    """
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by replacing zeros with NaN in specific columns and filling them.
    Args:
        df (pd.DataFrame): Raw data
    Returns:
        pd.DataFrame: Cleaned data
    """
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df

def preprocess_data(df: pd.DataFrame):
    """
    Split and scale the dataset.
    Args:
        df (pd.DataFrame): Cleaned data
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

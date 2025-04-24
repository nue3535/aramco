import pandas as pd
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(df):
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df


def preprocess_and_split(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def save_joblib(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(data, path)
    print(f"Saved: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to cleaned input CSV')
    parser.add_argument('--output_train_x', type=str, required=True)
    parser.add_argument('--output_test_x', type=str, required=True)
    parser.add_argument('--output_train_y', type=str, required=True)
    parser.add_argument('--output_test_y', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    save_joblib(X_train, args.output_train_x)
    save_joblib(X_test, args.output_test_x)
    save_joblib(y_train, args.output_train_y)
    save_joblib(y_test, args.output_test_y)

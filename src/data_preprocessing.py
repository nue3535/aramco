# Import required libraries
import pandas as pd
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from clearml import Task, StorageManager

# Function to clean the dataset by handling zero values
def clean_data(df):
    # Replace zero values with NaN for columns where 0 is not a valid physiological value
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    # Fill missing values with the median of each column
    df.fillna(df.median(), inplace=True)
    return df

# Function to preprocess data and split it into train and test sets
def preprocess_and_split(df):
    X = df.drop('Outcome', axis=1) # Features
    y = df['Outcome']              # Target

    # Stratified split to maintain class balance in training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Function to save data objects as .joblib files
def save_joblib(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(data, path)
    print(f"Saved: {path}")
    return path

# If input is a ClearML artifact URL, download it locally
def resolve_input_path(input_path):
    if input_path.startswith("http"):
        local_path = StorageManager.get_local_copy(remote_url=input_path)
        print(f"Downloaded input data to: {local_path}")
        return local_path
    return input_path

# Main function to execute preprocessing steps
def main(args):
    input_path = resolve_input_path(args['input_path'])
    df = pd.read_csv(input_path)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    # Save processed data to specified paths
    path_x_train = save_joblib(X_train, args['output_train_x'])
    path_x_test = save_joblib(X_test, args['output_test_x'])
    path_y_train = save_joblib(y_train, args['output_train_y'])
    path_y_test = save_joblib(y_test, args['output_test_y'])

    # Upload joblib files to ClearML for tracking
    task.upload_artifact("X_train", artifact_object=path_x_train)
    task.upload_artifact("X_test", artifact_object=path_x_test)
    task.upload_artifact("y_train", artifact_object=path_y_train)
    task.upload_artifact("y_test", artifact_object=path_y_test)

# Entry point for script execution
if __name__ == '__main__':
    # Initialize ClearML Task
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Data Preprocessing")
    task.execute_remotely()

    # Parse input/output paths from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/diabetes.csv')
    parser.add_argument('--output_train_x', type=str, default='outputs/X_train.joblib')
    parser.add_argument('--output_test_x', type=str, default='outputs/X_test.joblib')
    parser.add_argument('--output_train_y', type=str, default='outputs/y_train.joblib')
    parser.add_argument('--output_test_y', type=str, default='outputs/y_test.joblib')
    args = vars(parser.parse_args())

    # Run main function
    main(args)
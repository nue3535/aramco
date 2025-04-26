import pandas as pd
import argparse
import os
from clearml import Task

def load_raw_data(input_path):
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded data with shape: {df.shape}")
    return df

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")
    return output_path

def main(args):
    df = load_raw_data(args['input_path'])
    saved_path = save_data(df, args['output_path'])
    
    # Upload artifact to ClearML
    task.upload_artifact(name='cleaned_data_csv', artifact_object=saved_path)

if __name__ == '__main__':
    # Initialize ClearML task
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Data Ingestion")
    task.execute_remotely()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/diabetes.csv')
    parser.add_argument('--output_path', type=str, default='outputs/cleaned_data.csv')
    args = vars(parser.parse_args())

    main(args)
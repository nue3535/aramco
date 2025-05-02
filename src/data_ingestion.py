# Import required libraries
import pandas as pd
import argparse
import os
from clearml import Task

# Function to load the raw dataset from CSV
def load_raw_data(input_path):
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path) # Load data into DataFrame
    print(f"Loaded data with shape: {df.shape}")
    return df

# Function to save the dataset back to CSV
def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure the output directory exists
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")
    return output_path

# Main function to orchestrate the ingestion process
def main(args):
    # Step 1: Load the dataset
    df = load_raw_data(args['input_path'])

    # Step 2: Save it to output path
    saved_path = save_data(df, args['output_path'])
    
    # Step 3: Upload the saved CSV as an artifact to ClearML
    task.upload_artifact(name='cleaned_data_csv', artifact_object=saved_path)

if __name__ == '__main__':
    # Initialize the ClearML task with project and task name
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Data Ingestion")
    task.execute_remotely() # Runs task on ClearML agent if available

    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/diabetes.csv')
    parser.add_argument('--output_path', type=str, default='outputs/cleaned_data.csv')
    args = vars(parser.parse_args()) # Parse and convert arguments to dictionary

    # Run the main function with parsed arguments
    main(args)
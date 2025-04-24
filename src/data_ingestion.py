import pandas as pd
import argparse
import os


def load_raw_data(input_path):
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded data with shape: {df.shape}")
    return df


def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned data saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to raw input CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save cleaned output CSV')
    args = parser.parse_args()

    # Load and save the raw data without transformations
    df = load_raw_data(args.input_path)
    save_data(df, args.output_path)

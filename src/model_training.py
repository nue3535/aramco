# Import standard libraries
import argparse
import os
import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from clearml import Task, StorageManager

# Function to load best hyperparameters from JSON file
def load_best_params(model_type, path):
    if not path:
        return None
    path = StorageManager.get_local_copy(path)
    with open(path, 'r') as f:
        all_params = json.load(f)
    return all_params.get(model_type, {})

# Function to instantiate the selected model using provided parameters
def get_model(model_type, params):
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "svm":
        return SVC(probability=True, **params)
    else:
        raise ValueError("Unsupported model type.")

# Save trained model to disk using joblib
def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")
    return output_path

# Main training workflow
def main(args):
    # Download feature and label training data from ClearML storage
    path_x_train = StorageManager.get_local_copy(args['train_x'])
    path_y_train = StorageManager.get_local_copy(args['train_y'])

    # Load datasets
    X_train = joblib.load(path_x_train)
    y_train = joblib.load(path_y_train)

    # Load best hyperparameters
    best_params = load_best_params(args['model_type'], args.get('best_params_path'))
    print(f"Using best params for {args['model_type']}: {best_params}")

    # Create and train the model using loaded parameters
    model = get_model(args['model_type'], best_params)
    model.fit(X_train, y_train)

    # Save the trained model and upload as artifact to ClearML
    model_path = save_model(model, args['output_model'])
    task.upload_artifact(name=f"trained_model_{args['model_type']}", artifact_object=model_path)

if __name__ == '__main__':
    # Initialize ClearML task
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Training")
    task.execute_remotely()

    # Define and parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x', type=str, default='outputs/X_train.joblib')
    parser.add_argument('--train_y', type=str, default='outputs/y_train.joblib')
    parser.add_argument('--output_model', type=str, default='outputs/trained_model.joblib')
    parser.add_argument('--model_type', type=str, default='random_forest')
    parser.add_argument('--best_params_path', type=str, default=None)

    args = vars(parser.parse_args())
    main(args)

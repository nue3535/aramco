# Import required libraries
import argparse
import os
import json
import joblib
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from clearml import Task, StorageManager

# Function to run Optuna hyperparameter optimization
def run_optuna(model_name, X_train, y_train):
    # Split training data into training and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    def objective(trial):
        if model_name == "logistic_regression":
            # Suggest hyperparameters for logistic regression
            C = trial.suggest_float('C', 0.01, 10.0, log=True)
            solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
            model = LogisticRegression(C=C, solver=solver, max_iter=1000)

        elif model_name == "random_forest":
            # Suggest hyperparameters for random forest
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        elif model_name == "svm":
            # Suggest hyperparameters for support vector machine
            C = trial.suggest_float('C', 0.01, 10.0, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            model = SVC(C=C, kernel=kernel, probability=True)

        else:
            raise ValueError("Unsupported model name")

        # Train model and evaluate using recall score
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        return recall_score(y_val, y_pred)

    # Create and run the Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    # Return the best hyperparameters
    return study.best_params

# Main logic for hyperparameter tuning
def main(args):

    # Download training data from ClearML storage
    path_x_train = StorageManager.get_local_copy(args['train_x'])
    path_y_train = StorageManager.get_local_copy(args['train_y'])

    # Load training features and labels
    X_train = joblib.load(path_x_train)
    y_train = joblib.load(path_y_train)

    # Run Optuna for each model and collect best parameters
    best_params_dict = {}
    for model in ['logistic_regression', 'random_forest', 'svm']:
        print(f"Running HPO for {model}...")
        best_params = run_optuna(model, X_train, y_train)
        best_params_dict[model] = best_params
        print(f"Best parameters for {model}: {best_params}")

    # Save the best hyperparameters to JSON
    os.makedirs("outputs", exist_ok=True)
    best_param_path = "outputs/best_params.json"
    with open(best_param_path, "w") as f:
        json.dump(best_params_dict, f, indent=4)

    # Upload the best parameters as an artifact to ClearML
    task.upload_artifact("hpo_best_params", artifact_object=best_param_path)

if __name__ == "__main__":
    # Initialize the ClearML task
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - HPO Tuning")
    task.execute_remotely()

    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x', type=str, default='outputs/X_train.joblib')
    parser.add_argument('--train_y', type=str, default='outputs/y_train.joblib')
    args = vars(parser.parse_args())

    main(args)

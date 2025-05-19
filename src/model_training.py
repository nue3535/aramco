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

def load_best_params(model_type, path):
    if not path:
        return None
    path = StorageManager.get_local_copy(path)
    with open(path, 'r') as f:
        all_params = json.load(f)
    return all_params.get(model_type, {})

def get_model(model_type, params):
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "svm":
        return SVC(probability=True, **params)
    else:
        raise ValueError("Unsupported model type.")

def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")
    return output_path

def main(args):
    path_x_train = StorageManager.get_local_copy(args['train_x'])
    path_y_train = StorageManager.get_local_copy(args['train_y'])

    X_train = joblib.load(path_x_train)
    y_train = joblib.load(path_y_train)

    # Load best hyperparameters
    best_params = load_best_params(args['model_type'], args.get('best_params_path'))
    print(f"Using best params for {args['model_type']}: {best_params}")

    model = get_model(args['model_type'], best_params)
    model.fit(X_train, y_train)

    model_path = save_model(model, args['output_model'])
    task.upload_artifact(name=f"trained_model_{args['model_type']}", artifact_object=model_path)

if __name__ == '__main__':
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Training")
    task.execute_remotely()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x', type=str, default='outputs/X_train.joblib')
    parser.add_argument('--train_y', type=str, default='outputs/y_train.joblib')
    parser.add_argument('--output_model', type=str, default='outputs/trained_model.joblib')
    parser.add_argument('--model_type', type=str, default='random_forest')
    parser.add_argument('--best_params_path', type=str, default=None)

    args = vars(parser.parse_args())
    main(args)


## TODO

# # Import necessary libraries
# import argparse
# import os
# import joblib
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from clearml import Task, StorageManager

# # Function to train the model based on the specified type
# def train_model(model_name, X_train, y_train):
#     # Define model and its hyperparameter grid
#     if model_name == "logistic_regression":
#         model = LogisticRegression(max_iter=1000)
#         params = {'C': [0.1, 1.0, 10], 'solver': ['liblinear', 'lbfgs']}
#     elif model_name == "random_forest":
#         model = RandomForestClassifier(random_state=42)
#         params = {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
#     elif model_name == "svm":
#         model = SVC(probability=True)
#         params = {'C': [0.1, 1.0, 10], 'kernel': ['linear', 'rbf']}
#     else:
#         raise ValueError("Unsupported model name") # Error if model is not recognized

#     # Perform grid search to find the best hyperparameters
#     grid = GridSearchCV(model, params, cv=5)
#     grid.fit(X_train, y_train)
#     return grid.best_estimator_, grid.best_score_ # Return best model and score

# # Function to save the trained model
# def save_model(model, output_path):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     joblib.dump(model, output_path)
#     print(f"Model saved to: {output_path}")
#     return output_path

# # Main function to load data, train model, and save it
# def main(args):
#     # Download training data from ClearML if needed
#     path_x_train = StorageManager.get_local_copy(args['train_x'])
#     path_y_train = StorageManager.get_local_copy(args['train_y'])

#     # Load training feature and label data
#     X_train = joblib.load(path_x_train)
#     y_train = joblib.load(path_y_train)

#     # Train selected model type
#     model, score = train_model(args['model_type'], X_train, y_train)
#     print(f"Best CV Score for {args['model_type']}: {score:.4f}")

#     # Save the model and upload it to ClearML
#     model_path = save_model(model, args['output_model'])
#     task.upload_artifact(name=f"trained_model_{args['model_type']}", artifact_object=model_path)

# # Entry point when executed directly
# if __name__ == '__main__':
#     # Initialize ClearML task
#     task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Training")
#     task.execute_remotely() # Execute task on ClearML agent

#     # Parse command-line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_x', type=str, default='outputs/X_train.joblib')
#     parser.add_argument('--train_y', type=str, default='outputs/y_train.joblib')
#     parser.add_argument('--output_model', type=str, default='outputs/trained_model.joblib')
#     parser.add_argument('--model_type', type=str, default='random_forest') # Specify model type
    
#     args = vars(parser.parse_args())

#     # Run main pipeline
#     main(args)


## TODO



# # Import necessary libraries
# import argparse
# import os
# import joblib
# import optuna
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import recall_score
# from clearml import Task, StorageManager

# # Function to train the model using Optuna for HPO
# def train_model(model_name, X_train, y_train):
#     # Split training data into train and validation sets
#     X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

#     def objective(trial):
#         if model_name == "logistic_regression":
#             C = trial.suggest_loguniform('C', 0.01, 10.0)
#             solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
#             model = LogisticRegression(C=C, solver=solver, max_iter=1000)
#         elif model_name == "random_forest":
#             n_estimators = trial.suggest_int('n_estimators', 50, 200)
#             max_depth = trial.suggest_int('max_depth', 5, 30)
#             model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
#         elif model_name == "svm":
#             C = trial.suggest_loguniform('C', 0.01, 10.0)
#             kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
#             model = SVC(C=C, kernel=kernel, probability=True)
#         else:
#             raise ValueError("Unsupported model name.")

#         model.fit(X_tr, y_tr)
#         y_pred = model.predict(X_val)
#         recall = recall_score(y_val, y_pred)
#         return recall

#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=20)

#     print(f"Best trial parameters: {study.best_params}")

#     # Retrain the best model on the full training set
#     best_trial = study.best_trial
#     return objective(best_trial), study.best_trial

# # Save the trained model
# def save_model(model, output_path):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     joblib.dump(model, output_path)
#     print(f"Model saved to: {output_path}")
#     return output_path

# # Main function
# def main(args):
#     path_x_train = StorageManager.get_local_copy(args['train_x'])
#     path_y_train = StorageManager.get_local_copy(args['train_y'])

#     X_train = joblib.load(path_x_train)
#     y_train = joblib.load(path_y_train)

#     model, trial = train_model(args['model_type'], X_train, y_train)

#     print(f"Best Recall for {args['model_type']}: {trial.value:.4f}")
#     model_path = save_model(model, args['output_model'])
#     task.upload_artifact(name=f"trained_model_{args['model_type']}", artifact_object=model_path)

# # Script entry point
# if __name__ == '__main__':
#     task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Training")
#     task.execute_remotely()

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_x', type=str, default='outputs/X_train.joblib')
#     parser.add_argument('--train_y', type=str, default='outputs/y_train.joblib')
#     parser.add_argument('--output_model', type=str, default='outputs/trained_model.joblib')
#     parser.add_argument('--model_type', type=str, default='random_forest')

#     args = vars(parser.parse_args())
#     main(args)

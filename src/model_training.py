import argparse
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from clearml import Task

def train_model(model_name, X_train, y_train):
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
        params = {'C': [0.1, 1.0, 10], 'solver': ['liblinear', 'lbfgs']}
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
        params = {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
    elif model_name == "svm":
        model = SVC(probability=True)
        params = {'C': [0.1, 1.0, 10], 'kernel': ['linear', 'rbf']}
    else:
        raise ValueError("Unsupported model name")

    grid = GridSearchCV(model, params, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_

def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")
    return output_path

def main(args):
    X_train = joblib.load(args['train_x'])
    y_train = joblib.load(args['train_y'])

    model, score = train_model(args['model_type'], X_train, y_train)
    print(f"Best CV Score for {args['model_type']}: {score:.4f}")

    model_path = save_model(model, args['output_model'])
    task.upload_artifact(name=f"trained_model_{args['model_type']}", artifact_object=model_path)

if __name__ == '__main__':
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Training")
    # task.execute_remotely()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x', type=str, default='outputs/X_train.joblib')
    parser.add_argument('--train_y', type=str, default='outputs/y_train.joblib')
    parser.add_argument('--output_model', type=str, default='outputs/trained_model.joblib')
    parser.add_argument('--model_type', type=str, default='random_forest')
    args = vars(parser.parse_args())

    main(args)



##### TODO

# import argparse
# import os
# import joblib
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score


# def train_model(model_name, X_train, y_train):
#     if model_name == "logistic_regression":
#         model = LogisticRegression(max_iter=1000)
#         params = {'C': [0.1, 1.0, 10], 'solver': ['liblinear', 'lbfgs']}
#     elif model_name == "random_forest":
#         model = RandomForestClassifier(random_state=42)
#         params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
#     elif model_name == "svm":
#         model = SVC(probability=True)
#         params = {'C': [0.1, 1.0, 10], 'kernel': ['linear', 'rbf']}
#     else:
#         raise ValueError("Unsupported model name")

#     grid = GridSearchCV(model, params, cv=5)
#     grid.fit(X_train, y_train)
#     return grid.best_estimator_, grid.best_score_


# def save_model(model, output_path):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     joblib.dump(model, output_path)
#     print(f"Model saved to: {output_path}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_x', type=str, required=True)
#     parser.add_argument('--train_y', type=str, required=True)
#     parser.add_argument('--output_model', type=str, required=True)
#     parser.add_argument('--model_type', type=str, default="random_forest")
#     args = parser.parse_args()

#     X_train = joblib.load(args.train_x)
#     y_train = joblib.load(args.train_y)

#     model, score = train_model(args.model_type, X_train, y_train)
#     print(f"Best CV Score for {args.model_type}: {score:.4f}")

#     save_model(model, args.output_model)

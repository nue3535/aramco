import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression with hyperparameter tuning.
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"Best Logistic Regression Params: {grid.best_params_}")
    return grid.best_estimator_


def train_random_forest(X_train, y_train):
    """
    Random Forest with hyperparameter tuning.
    """
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"Best Random Forest Params: {grid.best_params_}")
    return grid.best_estimator_


def train_svm(X_train, y_train):
    """
    Support Vector Machine with hyperparameter tuning.
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"Best SVM Params: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy score.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred


def save_model(model, path="outputs/models/best_model.pkl"):
    """
    Save the trained model to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved at: {path}")

import argparse
import os
import joblib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use headless backend for plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from clearml import Task, StorageManager

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    return report, cm, roc_auc, y_prob

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved metrics to: {path}")
    return path

def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
    return save_path

def plot_roc_curve(y_test, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to: {save_path}")
    return save_path

def main(args):
    # Resolve remote artifact paths
    model_path = StorageManager.get_local_copy(args['model_path'])
    test_x_path = StorageManager.get_local_copy(args['test_x'])
    test_y_path = StorageManager.get_local_copy(args['test_y'])

    model = joblib.load(model_path)
    X_test = joblib.load(test_x_path)
    y_test = joblib.load(test_y_path)

    report, cm, roc_auc, y_prob = evaluate_model(model, X_test, y_test)

    metrics_path = save_json({"classification_report": report, "roc_auc": roc_auc}, args['output_metrics'])
    cm_path = plot_confusion_matrix(cm, labels=["Non-Diabetic", "Diabetic"], save_path=args['conf_matrix_path'])

    task.upload_artifact("metrics", artifact_object=metrics_path)
    task.upload_artifact("confusion_matrix", artifact_object=cm_path)

    if y_prob is not None and args.get('roc_curve_path'):
        roc_path = plot_roc_curve(y_test, y_prob, args['roc_curve_path'])
        task.upload_artifact("roc_curve", artifact_object=roc_path)

if __name__ == '__main__':
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Evaluation")
    # task.execute_remotely()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='outputs/trained_model.joblib') # , required=True TODO
    parser.add_argument('--test_x', type=str, default='outputs/X_test.joblib') # , required=True TODO
    parser.add_argument('--test_y', type=str, default='outputs/y_test.joblib') # , required=True TODO
    parser.add_argument('--output_metrics', type=str, default='outputs/metrics.json')
    parser.add_argument('--conf_matrix_path', type=str, default='outputs/confusion_matrix.png')
    parser.add_argument('--roc_curve_path', type=str, default='outputs/roc_curve.png')
    args = vars(parser.parse_args())

    main(args)



### TODO


# import argparse
# import matplotlib
# import os
# import joblib
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# from clearml import Task

# matplotlib.use('Agg')

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

#     report = classification_report(y_test, y_pred, output_dict=True)
#     cm = confusion_matrix(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

#     return report, cm, roc_auc, y_prob

# def save_json(data, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, 'w') as f:
#         json.dump(data, f, indent=4)
#     print(f"Saved metrics to: {path}")
#     return path

# def plot_confusion_matrix(cm, labels, save_path):
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Confusion matrix saved to: {save_path}")
#     return save_path

# def plot_roc_curve(y_test, y_prob, save_path):
#     fpr, tpr, _ = roc_curve(y_test, y_prob)
#     plt.figure()
#     plt.plot(fpr, tpr, label="ROC curve")
#     plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"ROC curve saved to: {save_path}")
#     return save_path

# def main(args):
#     model = joblib.load(args['model_path'])
#     X_test = joblib.load(args['test_x'])
#     y_test = joblib.load(args['test_y'])

#     report, cm, roc_auc, y_prob = evaluate_model(model, X_test, y_test)
#     metrics_path = save_json({"classification_report": report, "roc_auc": roc_auc}, args['output_metrics'])
#     cm_path = plot_confusion_matrix(cm, labels=["Non-Diabetic", "Diabetic"], save_path=args['conf_matrix_path'])

#     task.upload_artifact("metrics", artifact_object=metrics_path)
#     task.upload_artifact("confusion_matrix", artifact_object=cm_path)

#     if y_prob is not None and args.get('roc_curve_path'):
#         roc_path = plot_roc_curve(y_test, y_prob, args['roc_curve_path'])
#         task.upload_artifact("roc_curve", artifact_object=roc_path)

# if __name__ == '__main__':
#     task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Evaluation")
#     task.execute_remotely()

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default='outputs/trained_model.joblib')
#     parser.add_argument('--test_x', type=str, default='outputs/X_test.joblib')
#     parser.add_argument('--test_y', type=str, default='outputs/y_test.joblib')
#     parser.add_argument('--output_metrics', type=str, default='outputs/metrics.json')
#     parser.add_argument('--conf_matrix_path', type=str, default='outputs/confusion_matrix.png')
#     parser.add_argument('--roc_curve_path', type=str, default='outputs/roc_curve.png')
#     args = vars(parser.parse_args())

#     main(args)



### TODO

# import argparse
# import os
# import joblib
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

#     report = classification_report(y_test, y_pred, output_dict=True)
#     cm = confusion_matrix(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

#     return report, cm, roc_auc, y_prob


# def save_json(data, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, 'w') as f:
#         json.dump(data, f, indent=4)
#     print(f"Saved metrics to: {path}")


# def plot_confusion_matrix(cm, labels, save_path):
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Confusion matrix saved to: {save_path}")


# def plot_roc_curve(y_test, y_prob, save_path):
#     fpr, tpr, _ = roc_curve(y_test, y_prob)
#     plt.figure()
#     plt.plot(fpr, tpr, label="ROC curve")
#     plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"ROC curve saved to: {save_path}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--test_x', type=str, required=True)
#     parser.add_argument('--test_y', type=str, required=True)
#     parser.add_argument('--output_metrics', type=str, required=True)
#     parser.add_argument('--conf_matrix_path', type=str, required=True)
#     parser.add_argument('--roc_curve_path', type=str, required=False)
#     args = parser.parse_args()

#     model = joblib.load(args.model_path)
#     X_test = joblib.load(args.test_x)
#     y_test = joblib.load(args.test_y)

#     report, cm, roc_auc, y_prob = evaluate_model(model, X_test, y_test)
#     metrics = {
#         "classification_report": report,
#         "roc_auc": roc_auc
#     }

#     save_json(metrics, args.output_metrics)
#     plot_confusion_matrix(cm, labels=["Non-Diabetic", "Diabetic"], save_path=args.conf_matrix_path)

#     if y_prob is not None and args.roc_curve_path:
#         plot_roc_curve(y_test, y_prob, args.roc_curve_path)

import argparse
import os
import json
import shutil
from clearml import Task

def load_metrics(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def select_model_by_roc_auc(model_path, metrics_path, selected_model_path):
    metrics = load_metrics(metrics_path)
    roc_auc = metrics.get("roc_auc", None)

    if roc_auc is not None and roc_auc > 0.7:
        os.makedirs(os.path.dirname(selected_model_path), exist_ok=True)
        shutil.copy(model_path, selected_model_path)
        print(f"Model selected (ROC-AUC={roc_auc:.2f}) and saved to: {selected_model_path}")
        task.upload_artifact("selected_model", artifact_object=selected_model_path)
    else:
        print(f"ROC-AUC too low ({roc_auc}), model not selected.")

def main(args):
    select_model_by_roc_auc(args['model_path'], args['metrics_path'], args['selected_model_path'])

if __name__ == '__main__':
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Selection")
    task.execute_remotely()

    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_path', type=str, default='outputs/metrics.json')
    parser.add_argument('--model_path', type=str, default='outputs/trained_model.joblib')
    parser.add_argument('--selected_model_path', type=str, default='outputs/best_model.joblib')
    args = vars(parser.parse_args())

    main(args)

### TODO

# import argparse
# import os
# import json
# import shutil


# def load_metrics(metrics_path):
#     with open(metrics_path, 'r') as f:
#         metrics = json.load(f)
#     return metrics


# def select_model_by_roc_auc(model_path, metrics_path, selected_model_path):
#     metrics = load_metrics(metrics_path)
#     roc_auc = metrics.get("roc_auc", None)

#     if roc_auc is not None and roc_auc > 0.7:
#         os.makedirs(os.path.dirname(selected_model_path), exist_ok=True)
#         shutil.copy(model_path, selected_model_path)
#         print(f"Model selected based on ROC-AUC ({roc_auc:.2f}) and saved to: {selected_model_path}")
#     else:
#         print(f"Model ROC-AUC too low ({roc_auc}). Not selected for release.")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--metrics_path', type=str, required=True)
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--selected_model_path', type=str, required=True)
#     args = parser.parse_args()

#     select_model_by_roc_auc(args.model_path, args.metrics_path, args.selected_model_path)

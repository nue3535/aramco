import argparse
import os
import json
import shutil
from clearml import Task

def load_metrics(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def main(args):
    task = Task.current_task()

    try:
        eval_rf = Task.get_task(task_id=args['eval_rf_id'])
        eval_svm = Task.get_task(task_id=args['eval_svm_id'])
        eval_logreg = Task.get_task(task_id=args['eval_logreg_id'])

        rf_metrics_path = eval_rf.artifacts['metrics'].get_local_copy()
        svm_metrics_path = eval_svm.artifacts['metrics'].get_local_copy()
        logreg_metrics_path = eval_logreg.artifacts['metrics'].get_local_copy()

        rf_metrics = load_metrics(rf_metrics_path)
        svm_metrics = load_metrics(svm_metrics_path)
        logreg_metrics = load_metrics(logreg_metrics_path)

        rf_auc = rf_metrics.get('roc_auc', 0)
        svm_auc = svm_metrics.get('roc_auc', 0)
        logreg_auc = logreg_metrics.get('roc_auc', 0)

        print(f"Random Forest ROC AUC: {rf_auc:.4f}")
        print(f"SVM ROC AUC: {svm_auc:.4f}")
        print(f"Logistic Regression ROC AUC: {logreg_auc:.4f}")

        scores = {
            "Random Forest": rf_auc,
            "SVM": svm_auc,
            "Logistic Regression": logreg_auc
        }

        best_model = max(scores, key=scores.get)
        print(f"Best Model Selected: {best_model}")

        os.makedirs("outputs", exist_ok=True)
        best_model_path = "outputs/best_model.txt"
        with open(best_model_path, 'w') as f:
            f.write(f"Best Model: {best_model}")
        task.upload_artifact("best_model_name", artifact_object=best_model_path)

    except Exception as e:
        print("Skipped model selection logic during registration (no real tasks yet)")
        pass

if __name__ == '__main__':
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Selection")
    # task.execute_remotely()

    # During template registration, use placeholder values
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_rf_id', type=str, default="rf_dummy_id")
    parser.add_argument('--eval_svm_id', type=str, default="svm_dummy_id")
    parser.add_argument('--eval_logreg_id', type=str, default="logreg_dummy_id")
    args = vars(parser.parse_args())

    main(args)


#TODO

# import argparse
# import os
# import json
# import shutil
# from clearml import Task

# def load_metrics(metrics_path):
#     with open(metrics_path, 'r') as f:
#         metrics = json.load(f)
#     return metrics

# def main(args):
#     task = Task.current_task()

#     # Fetch evaluation tasks
#     eval_rf = Task.get_task(task_id=args['eval_rf_id'])
#     eval_svm = Task.get_task(task_id=args['eval_svm_id'])
#     eval_logreg = Task.get_task(task_id=args['eval_logreg_id'])

#     # Download evaluation artifacts (metrics.json) from each evaluation step
#     rf_metrics_path = eval_rf.artifacts['metrics'].get_local_copy()
#     svm_metrics_path = eval_svm.artifacts['metrics'].get_local_copy()
#     logreg_metrics_path = eval_logreg.artifacts['metrics'].get_local_copy()

#     # Load ROC AUC scores
#     rf_metrics = load_metrics(rf_metrics_path)
#     svm_metrics = load_metrics(svm_metrics_path)
#     logreg_metrics = load_metrics(logreg_metrics_path)

#     rf_auc = rf_metrics.get('roc_auc', 0)
#     svm_auc = svm_metrics.get('roc_auc', 0)
#     logreg_auc = logreg_metrics.get('roc_auc', 0)

#     print(f"Random Forest ROC AUC: {rf_auc:.4f}")
#     print(f"SVM ROC AUC: {svm_auc:.4f}")
#     print(f"Logistic Regression ROC AUC: {logreg_auc:.4f}")

#     # Compare and pick the best model based on ROC AUC
#     scores = {
#         "Random Forest": rf_auc,
#         "SVM": svm_auc,
#         "Logistic Regression": logreg_auc
#     }

#     best_model = max(scores, key=scores.get)
#     print(f"Best Model Selected: {best_model}")

#     # Save the result
#     output_dir = "outputs"
#     os.makedirs(output_dir, exist_ok=True)
#     best_model_path = os.path.join(output_dir, "best_model.txt")
#     with open(best_model_path, 'w') as f:
#         f.write(f"Best Model: {best_model}")

#     # Upload the best model name as an artifact
#     task.upload_artifact(name="best_model_name", artifact_object=best_model_path)

# if __name__ == '__main__':
#     task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Selection")
#     # task.execute_remotely()

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--eval_rf_id', type=str)
#     parser.add_argument('--eval_svm_id', type=str)
#     parser.add_argument('--eval_logreg_id', type=str)
#     args = vars(parser.parse_args())

#     main(args)


## TODO

# import argparse
# import os
# import json
# import shutil
# from clearml import Task

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
#         print(f"Model selected (ROC-AUC={roc_auc:.2f}) and saved to: {selected_model_path}")
#         task.upload_artifact("selected_model", artifact_object=selected_model_path)
#     else:
#         print(f"ROC-AUC too low ({roc_auc}), model not selected.")

# def main(args):
#     select_model_by_roc_auc(args['model_path'], args['metrics_path'], args['selected_model_path'])

# if __name__ == '__main__':
#     task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Selection")
#     task.execute_remotely()

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--metrics_path', type=str, default='outputs/metrics.json')
#     parser.add_argument('--model_path', type=str, default='outputs/trained_model.joblib')
#     parser.add_argument('--selected_model_path', type=str, default='outputs/best_model.joblib')
#     args = vars(parser.parse_args())

#     main(args)

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

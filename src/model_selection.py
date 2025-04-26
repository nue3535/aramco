import argparse
import os
import json
import shutil
from clearml import Task

def load_metrics(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def extract_recall_class1(metrics):
    try:
        return metrics['classification_report']['1']['recall']
    except KeyError:
        return 0.0

def main(args):
    task = Task.current_task()

    try:
        # Fetch evaluation tasks
        eval_rf = Task.get_task(task_id=args['eval_rf_id'])
        eval_svm = Task.get_task(task_id=args['eval_svm_id'])
        eval_logreg = Task.get_task(task_id=args['eval_logreg_id'])

        # Download metrics artifacts
        rf_metrics_path = eval_rf.artifacts['metrics'].get_local_copy()
        svm_metrics_path = eval_svm.artifacts['metrics'].get_local_copy()
        logreg_metrics_path = eval_logreg.artifacts['metrics'].get_local_copy()

        # Load metrics
        rf_metrics = load_metrics(rf_metrics_path)
        svm_metrics = load_metrics(svm_metrics_path)
        logreg_metrics = load_metrics(logreg_metrics_path)

        # Extract recall for class '1' (diabetic)
        rf_recall = extract_recall_class1(rf_metrics)
        svm_recall = extract_recall_class1(svm_metrics)
        logreg_recall = extract_recall_class1(logreg_metrics)

        print(f"Random Forest Recall (Class 1): {rf_recall:.4f}")
        print(f"SVM Recall (Class 1): {svm_recall:.4f}")
        print(f"Logistic Regression Recall (Class 1): {logreg_recall:.4f}")

        scores = {
            "Random Forest": rf_recall,
            "SVM": svm_recall,
            "Logistic Regression": logreg_recall
        }

        # Select model with highest recall
        best_model = max(scores, key=scores.get)
        print(f"Best Model Selected (by Recall of Class 1): {best_model}")

        # Save and upload result
        os.makedirs("outputs", exist_ok=True)
        best_model_path = "outputs/best_model.txt"
        with open(best_model_path, 'w') as f:
            f.write(f"Best Model: {best_model}")
        task.upload_artifact("best_model_name", artifact_object=best_model_path)

    except Exception as e:
        print("Skipped model selection logic during registration. Reason:", str(e))
        pass

if __name__ == '__main__':
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Selection")
    task.execute_remotely()

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

#     try:
#         eval_rf = Task.get_task(task_id=args['eval_rf_id'])
#         eval_svm = Task.get_task(task_id=args['eval_svm_id'])
#         eval_logreg = Task.get_task(task_id=args['eval_logreg_id'])

#         rf_metrics_path = eval_rf.artifacts['metrics'].get_local_copy()
#         svm_metrics_path = eval_svm.artifacts['metrics'].get_local_copy()
#         logreg_metrics_path = eval_logreg.artifacts['metrics'].get_local_copy()

#         rf_metrics = load_metrics(rf_metrics_path)
#         svm_metrics = load_metrics(svm_metrics_path)
#         logreg_metrics = load_metrics(logreg_metrics_path)

#         rf_auc = rf_metrics.get('roc_auc', 0)
#         svm_auc = svm_metrics.get('roc_auc', 0)
#         logreg_auc = logreg_metrics.get('roc_auc', 0)

#         print(f"Random Forest ROC AUC: {rf_auc:.4f}")
#         print(f"SVM ROC AUC: {svm_auc:.4f}")
#         print(f"Logistic Regression ROC AUC: {logreg_auc:.4f}")

#         scores = {
#             "Random Forest": rf_auc,
#             "SVM": svm_auc,
#             "Logistic Regression": logreg_auc
#         }

#         best_model = max(scores, key=scores.get)
#         print(f"Best Model Selected: {best_model}")

#         os.makedirs("outputs", exist_ok=True)
#         best_model_path = "outputs/best_model.txt"
#         with open(best_model_path, 'w') as f:
#             f.write(f"Best Model: {best_model}")
#         task.upload_artifact("best_model_name", artifact_object=best_model_path)

#     except Exception as e:
#         print("Skipped model selection logic during registration (no real tasks yet)")
#         pass

# if __name__ == '__main__':
#     task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Selection")
#     task.execute_remotely()

#     # During template registration, use placeholder values
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--eval_rf_id', type=str, default="rf_dummy_id")
#     parser.add_argument('--eval_svm_id', type=str, default="svm_dummy_id")
#     parser.add_argument('--eval_logreg_id', type=str, default="logreg_dummy_id")
#     args = vars(parser.parse_args())

#     main(args)

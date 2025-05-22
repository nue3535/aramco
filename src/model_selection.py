# Import required libraries
import argparse
import os
import json
# import shutil #TODO
from clearml import Task

# Function to load JSON-formatted evaluation metrics
def load_metrics(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

# Function to extract recall for class '1' (diabetic) from classification report
def extract_recall_class1(metrics):
    try:
        return metrics['classification_report']['1']['recall']
    except KeyError:
        return 0.0 # Return zero if recall is missing

# Main logic for model selection
def main(args):
    task = Task.current_task() # Get current ClearML task context

    try:
        # Retrieve evaluation tasks using ClearML Task IDs
        eval_rf = Task.get_task(task_id=args['eval_rf_id'])
        eval_svm = Task.get_task(task_id=args['eval_svm_id'])
        eval_logreg = Task.get_task(task_id=args['eval_logreg_id'])

        # Download metric artifact files from each evaluation task
        rf_metrics_path = eval_rf.artifacts['metrics'].get_local_copy()
        svm_metrics_path = eval_svm.artifacts['metrics'].get_local_copy()
        logreg_metrics_path = eval_logreg.artifacts['metrics'].get_local_copy()

        # Load evaluation metrics from JSON
        rf_metrics = load_metrics(rf_metrics_path)
        svm_metrics = load_metrics(svm_metrics_path)
        logreg_metrics = load_metrics(logreg_metrics_path)

        # Extract recall values for class 1 (positive diabetes diagnosis)
        rf_recall = extract_recall_class1(rf_metrics)
        svm_recall = extract_recall_class1(svm_metrics)
        logreg_recall = extract_recall_class1(logreg_metrics)

        # Print recall scores
        print(f"Random Forest Recall (Class 1): {rf_recall:.4f}")
        print(f"SVM Recall (Class 1): {svm_recall:.4f}")
        print(f"Logistic Regression Recall (Class 1): {logreg_recall:.4f}")

        # Create a dictionary to store all recall scores
        scores = {
            "Random Forest": rf_recall,
            "SVM": svm_recall,
            "Logistic Regression": logreg_recall
        }

        # Select the model with the highest recall
        best_model = max(scores, key=scores.get)
        print(f"Best Model Selected (by Recall of Class 1): {best_model}")

        # Save the selected model name to a text file
        os.makedirs("outputs", exist_ok=True)
        best_model_path = "outputs/best_model.txt"
        with open(best_model_path, 'w') as f:
            f.write(f"Best Model: {best_model}")

        # Upload best model name as an artifact
        task.upload_artifact("best_model_name", artifact_object=best_model_path)

        #Tag best model artifact
        training_task_ids = {
            "Random Forest": args['train_rf_id'],
            "SVM": args['train_svm_id'],
            "Logistic Regression": args['train_logreg_id']
        }

        best_training_task = Task.get_task(task_id=training_task_ids[best_model])
        artifact_key = f"trained_model_{best_model.lower().replace(' ', '_')}"

        if artifact_key in best_training_task.artifacts:
            artifact = best_training_task.artifacts[artifact_key]
            artifact.metadata['tag'] = 'best_production_model'
            best_training_task.set_artifacts({artifact_key: artifact})
            print(f"Tagged '{artifact_key}' as 'best_production_model'")
        else:
            print(f"Artifact '{artifact_key}' not found in task '{best_model}'")

    except Exception as e:
        # Fail-safe for early pipeline registration
        print("Skipped model selection logic during registration. Reason:", str(e))
        pass

# Entry point for ClearML task
if __name__ == '__main__':
    # Initialize ClearML task for model selection
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Selection")
    task.execute_remotely()

    # Parse evaluation task IDs passed from the pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_rf_id', type=str, default="rf_dummy_id")
    parser.add_argument('--eval_svm_id', type=str, default="svm_dummy_id")
    parser.add_argument('--eval_logreg_id', type=str, default="logreg_dummy_id")
    parser.add_argument('--train_rf_id', type=str, default="rf_train_dummy_id")
    parser.add_argument('--train_svm_id', type=str, default="svm_train_dummy_id")
    parser.add_argument('--train_logreg_id', type=str, default="logreg_train_dummy_id")

    args = vars(parser.parse_args())

    # Run the model selection logic
    main(args)


#TODO


# # Import required libraries
# import argparse
# import os
# import json
# # import shutil #TODO
# from clearml import Task

# # Function to load JSON-formatted evaluation metrics
# def load_metrics(metrics_path):
#     with open(metrics_path, 'r') as f:
#         metrics = json.load(f)
#     return metrics

# # Function to extract recall for class '1' (diabetic) from classification report
# def extract_recall_class1(metrics):
#     try:
#         return metrics['classification_report']['1']['recall']
#     except KeyError:
#         return 0.0 # Return zero if recall is missing

# # Main logic for model selection
# def main(args):
#     task = Task.current_task() # Get current ClearML task context

#     try:
#         # Retrieve evaluation tasks using ClearML Task IDs
#         eval_rf = Task.get_task(task_id=args['eval_rf_id'])
#         eval_svm = Task.get_task(task_id=args['eval_svm_id'])
#         eval_logreg = Task.get_task(task_id=args['eval_logreg_id'])

#         # Download metric artifact files from each evaluation task
#         rf_metrics_path = eval_rf.artifacts['metrics'].get_local_copy()
#         svm_metrics_path = eval_svm.artifacts['metrics'].get_local_copy()
#         logreg_metrics_path = eval_logreg.artifacts['metrics'].get_local_copy()

#         # Load evaluation metrics from JSON
#         rf_metrics = load_metrics(rf_metrics_path)
#         svm_metrics = load_metrics(svm_metrics_path)
#         logreg_metrics = load_metrics(logreg_metrics_path)

#         # Extract recall values for class 1 (positive diabetes diagnosis)
#         rf_recall = extract_recall_class1(rf_metrics)
#         svm_recall = extract_recall_class1(svm_metrics)
#         logreg_recall = extract_recall_class1(logreg_metrics)

#         # Print recall scores
#         print(f"Random Forest Recall (Class 1): {rf_recall:.4f}")
#         print(f"SVM Recall (Class 1): {svm_recall:.4f}")
#         print(f"Logistic Regression Recall (Class 1): {logreg_recall:.4f}")

#         # Create a dictionary to store all recall scores
#         scores = {
#             "Random Forest": rf_recall,
#             "SVM": svm_recall,
#             "Logistic Regression": logreg_recall
#         }

#         # Select the model with the highest recall
#         best_model = max(scores, key=scores.get)
#         print(f"Best Model Selected (by Recall of Class 1): {best_model}")

#         # Save the selected model name to a text file
#         os.makedirs("outputs", exist_ok=True)
#         best_model_path = "outputs/best_model.txt"
#         with open(best_model_path, 'w') as f:
#             f.write(f"Best Model: {best_model}")

#         # Upload best model name as an artifact
#         task.upload_artifact("best_model_name", artifact_object=best_model_path)

#     except Exception as e:
#         # Fail-safe for early pipeline registration
#         print("Skipped model selection logic during registration. Reason:", str(e))
#         pass

# # Entry point for ClearML task
# if __name__ == '__main__':
#     # Initialize ClearML task for model selection
#     task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Model Selection")
#     task.execute_remotely()

#     # Parse evaluation task IDs passed from the pipeline
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--eval_rf_id', type=str, default="rf_dummy_id")
#     parser.add_argument('--eval_svm_id', type=str, default="svm_dummy_id")
#     parser.add_argument('--eval_logreg_id', type=str, default="logreg_dummy_id")
#     parser.add_argument('--train_rf_id', type=str, default="rf_train_dummy_id")
#     parser.add_argument('--train_svm_id', type=str, default="svm_train_dummy_id")
#     parser.add_argument('--train_logreg_id', type=str, default="logreg_train_dummy_id")

#     args = vars(parser.parse_args())

#     # Run the model selection logic
#     main(args)

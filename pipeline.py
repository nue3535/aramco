# Import the ClearML PipelineController to define and run automated pipelines
from clearml.automation import PipelineController

# Define the main function to create and run the pipeline
def run_pipeline():
    pipe = PipelineController(
        name="Diabetes Prediction Pipeline", # Name of the pipeline
        project="AI for Diabetes Prediction", # Associated ClearML project
        version="1.0" # Versioning for pipeline tracking
    )

    # Set the default ClearML execution queue where all steps will be executed
    pipe.set_default_execution_queue("default")

    # Data Ingestion
    # This step reads the raw dataset and saves a cleaned CSV
    pipe.add_step(
        name="data_ingestion",
        base_task_project="AI for Diabetes Prediction", # Project where the template task is stored
        base_task_name="Template - Data Ingestion", # Reusable task template
        parameter_override={
            "Args/input_path": "data/diabetes.csv", # Input file path
            "Args/output_path": "outputs/cleaned_data.csv" # Output cleaned CSV file
        }
    )

    # Data Preprocessing
    # This step cleans, scales, and splits the data into training/testing sets
    pipe.add_step(
        name="data_preprocessing",
        parents=["data_ingestion"], # This step depends on data_ingestion output
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Data Preprocessing",
        parameter_override={
            # Use output from data_ingestion as input
            "Args/input_path": "${data_ingestion.artifacts.cleaned_data_csv.url}",
            "Args/output_train_x": "outputs/X_train.joblib",
            "Args/output_test_x": "outputs/X_test.joblib",
            "Args/output_train_y": "outputs/y_train.joblib",
            "Args/output_test_y": "outputs/y_test.joblib"
        }
    )

    # HPO Tuning
    pipe.add_step(
        name="hpo_tuning",
        parents=["data_preprocessing"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - HPO Tuning",
        parameter_override={
            "Args/train_x": "${data_preprocessing.artifacts.X_train.url}",
            "Args/train_y": "${data_preprocessing.artifacts.y_train.url}"
        }
    )

    # Model Training (Random Forest)
    pipe.add_step(
        name="train_rf",
        parents=["hpo_tuning"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Training",
        parameter_override={
            "Args/train_x": "${data_preprocessing.artifacts.X_train.url}",
            "Args/train_y": "${data_preprocessing.artifacts.y_train.url}",
            "Args/output_model": "outputs/rf_model.joblib",
            "Args/best_params_path": "${hpo_tuning.artifacts.hpo_best_params.url}", # Output model file
            "Args/model_type": "random_forest" # Specify model type
        }
    )

    # Model Training (SVM)
    pipe.add_step(
        name="train_svm",
        parents=["hpo_tuning"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Training",
        parameter_override={
            "Args/train_x": "${data_preprocessing.artifacts.X_train.url}",
            "Args/train_y": "${data_preprocessing.artifacts.y_train.url}",
            "Args/output_model": "outputs/svm_model.joblib",
            "Args/best_params_path": "${hpo_tuning.artifacts.hpo_best_params.url}",
            "Args/model_type": "svm"
        }
    )

    # Model Training (Logistic Regression)
    pipe.add_step(
        name="train_logreg",
        parents=["hpo_tuning"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Training",
        parameter_override={
            "Args/train_x": "${data_preprocessing.artifacts.X_train.url}",
            "Args/train_y": "${data_preprocessing.artifacts.y_train.url}",
            "Args/output_model": "outputs/logreg_model.joblib",
            "Args/best_params_path": "${hpo_tuning.artifacts.hpo_best_params.url}",
            "Args/model_type": "logistic_regression"
        }
    )

    # Model Evaluation (Random Forest)
    pipe.add_step(
        name="eval_rf",
        parents=["train_rf"], # Depends on RF model being trained
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Evaluation", # Reusable evaluation task
        parameter_override={
            "Args/model_path": "${train_rf.artifacts.trained_model_random_forest.url}", # Use trained RF model
            "Args/test_x": "${data_preprocessing.artifacts.X_test.url}", # Test features
            "Args/test_y": "${data_preprocessing.artifacts.y_test.url}" # Test labels
        }
    )

    # Model Evaluation (SVM)
    pipe.add_step(
        name="eval_svm",
        parents=["train_svm"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Evaluation",
        parameter_override={
            "Args/model_path": "${train_svm.artifacts.trained_model_svm.url}",
            "Args/test_x": "${data_preprocessing.artifacts.X_test.url}",
            "Args/test_y": "${data_preprocessing.artifacts.y_test.url}"
        }
    )

    # Model Evaluation (Logistic Regression)
    pipe.add_step(
        name="eval_logreg",
        parents=["train_logreg"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Evaluation",
        parameter_override={
            "Args/model_path": "${train_logreg.artifacts.trained_model_logistic_regression.url}",
            "Args/test_x": "${data_preprocessing.artifacts.X_test.url}",
            "Args/test_y": "${data_preprocessing.artifacts.y_test.url}"
        }
    )

    # Model Selection
    # Compares evaluation results from all models and selects the best one
    pipe.add_step(
        name="model_selection",
        parents=["eval_rf", "eval_svm", "eval_logreg"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Selection",
        parameter_override={
            "Args/eval_rf_id": "${eval_rf.id}", # Pass RF evaluation task ID
            "Args/eval_svm_id": "${eval_svm.id}", # Pass SVM evaluation task ID
            "Args/eval_logreg_id": "${eval_logreg.id}" # Pass LogReg evaluation task ID
            "Args/train_rf_id": "${train_rf.id}", # Pass RF training task ID
            "Args/train_svm_id": "${train_svm.id}", # Pass SVM training task ID
            "Args/train_logreg_id": "${train_logreg.id}", # Pass LogReg training task ID

        }
    )

    # Start the pipeline
    pipe.start_locally()
    print("Pipeline execution complete!")

if __name__ == "__main__":
    run_pipeline()
    
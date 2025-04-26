from clearml.automation import PipelineController

def run_pipeline():
    pipe = PipelineController(
        name="Diabetes Prediction Pipeline",
        project="AI for Diabetes Prediction",
        version="1.0"
    )

    pipe.set_default_execution_queue("default")

    # 1. Data Ingestion
    pipe.add_step(
        name="data_ingestion",
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Data Ingestion",
        parameter_override={
            "Args/input_path": "data/diabetes.csv",
            "Args/output_path": "outputs/cleaned_data.csv"
        }
    )

    # 2. Data Preprocessing
    pipe.add_step(
        name="data_preprocessing",
        parents=["data_ingestion"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Data Preprocessing",
        parameter_override={
            "Args/input_path": "${data_ingestion.artifacts.cleaned_data_csv.url}",
            "Args/output_train_x": "outputs/X_train.joblib",
            "Args/output_test_x": "outputs/X_test.joblib",
            "Args/output_train_y": "outputs/y_train.joblib",
            "Args/output_test_y": "outputs/y_test.joblib"
        }
    )

    # 3. Model Training (Random Forest)
    pipe.add_step(
        name="train_rf",
        parents=["data_preprocessing"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Training",
        parameter_override={
            "Args/train_x": "${data_preprocessing.artifacts.X_train.url}",
            "Args/train_y": "${data_preprocessing.artifacts.y_train.url}",
            "Args/output_model": "outputs/rf_model.joblib",
            "Args/model_type": "random_forest"
        }
    )

    # 4. Model Training (SVM)
    pipe.add_step(
        name="train_svm",
        parents=["data_preprocessing"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Training",
        parameter_override={
            "Args/train_x": "${data_preprocessing.artifacts.X_train.url}",
            "Args/train_y": "${data_preprocessing.artifacts.y_train.url}",
            "Args/output_model": "outputs/svm_model.joblib",
            "Args/model_type": "svm"
        }
    )

    # 5. Model Training (Logistic Regression)
    pipe.add_step(
        name="train_logreg",
        parents=["data_preprocessing"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Training",
        parameter_override={
            "Args/train_x": "${data_preprocessing.artifacts.X_train.url}",
            "Args/train_y": "${data_preprocessing.artifacts.y_train.url}",
            "Args/output_model": "outputs/logreg_model.joblib",
            "Args/model_type": "logistic_regression"
        }
    )

    # 6. Model Evaluation (Random Forest)
    pipe.add_step(
        name="eval_rf",
        parents=["train_rf"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Evaluation",
        parameter_override={
            "Args/model_path": "${train_rf.artifacts.trained_model_random_forest.url}",
            "Args/test_x": "${data_preprocessing.artifacts.X_test.url}",
            "Args/test_y": "${data_preprocessing.artifacts.y_test.url}"
        }
    )

    # 7. Model Evaluation (SVM)
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

    # 8. Model Evaluation (Logistic Regression)
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

    # 9. Model Selection
    pipe.add_step(
        name="model_selection",
        parents=["eval_rf", "eval_svm", "eval_logreg"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Selection",
        parameter_override={
            "Args/eval_rf_id": "${eval_rf.id}",
            "Args/eval_svm_id": "${eval_svm.id}",
            "Args/eval_logreg_id": "${eval_logreg.id}"
        }
    )

    # Start the pipeline
    pipe.start_locally()
    print("Pipeline execution complete!")

if __name__ == "__main__":
    run_pipeline()

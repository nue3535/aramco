from clearml.automation import PipelineController

def run_pipeline():
    pipe = PipelineController(
        name="Diabetes Prediction Pipeline",
        project="AI for Diabetes Prediction",
        version="1.0"
    )

    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="data_ingestion",
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Data Ingestion",
        parameter_override={
            "Args/input_path": "data/diabetes.csv",
            "Args/output_path": "outputs/cleaned_data.csv"
        }
    )

    pipe.add_step(
        name="data_preprocessing",
        parents=["data_ingestion"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Data Preprocessing",
        parameter_override={
            "Args/input_path": "outputs/cleaned_data.csv",
            "Args/output_train_x": "outputs/X_train.joblib",
            "Args/output_test_x": "outputs/X_test.joblib",
            "Args/output_train_y": "outputs/y_train.joblib",
            "Args/output_test_y": "outputs/y_test.joblib"
        }
    )

    pipe.add_step(
        name="model_training",
        parents=["data_preprocessing"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Training",
        parameter_override={
            "Args/train_x": "outputs/X_train.joblib",
            "Args/train_y": "outputs/y_train.joblib",
            "Args/output_model": "outputs/trained_model.joblib",
            "Args/model_type": "random_forest"
        }
    )

    pipe.add_step(
        name="model_evaluation",
        parents=["model_training"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Evaluation",
        parameter_override={
            "Args/model_path": "outputs/trained_model.joblib",
            "Args/test_x": "outputs/X_test.joblib",
            "Args/test_y": "outputs/y_test.joblib",
            "Args/output_metrics": "outputs/metrics.json",
            "Args/conf_matrix_path": "outputs/confusion_matrix.png",
            "Args/roc_curve_path": "outputs/roc_curve.png"
        }
    )

    pipe.add_step(
        name="model_selection",
        parents=["model_evaluation"],
        base_task_project="AI for Diabetes Prediction",
        base_task_name="Template - Model Selection",
        parameter_override={
            "Args/metrics_path": "outputs/metrics.json",
            "Args/model_path": "outputs/trained_model.joblib",
            "Args/selected_model_path": "outputs/best_model.joblib"
        }
    )

    pipe.start()
    pipe.wait()
    print("Pipeline execution complete.")

if __name__ == "__main__":
    run_pipeline()


## TODO


# from clearml import Task
# from clearml.automation import PipelineController

# # Initialize the ClearML Task
# pipeline = PipelineController(
#     name="Diabetes Prediction Pipeline",
#     project="AI for Diabetes Prediction",
#     version="0.1",
#     add_pipeline_tags=False
# )

# pipeline.set_default_execution_queue("default")

# # Step 1: Data Ingestion
# pipeline.add_step(
#     name="data_ingestion",
#     base_task_project="AI for Diabetes Prediction",
#     base_task_name="Data Ingestion",
#     # script="src/data_ingestion.py",
#     # working_directory=".",
#     # docker=None,
#     arguments={
#         "--input_path": "data/diabetes.csv",
#         "--output_path": "outputs/cleaned_data.csv"
#     },
#     task_type=Task.TaskTypes.data_processing
# )

# # Step 2: Data Preprocessing
# pipeline.add_step(
#     name="data_preprocessing",
#     parents=["data_ingestion"],
#     base_task_project="AI for Diabetes Prediction",
#     base_task_name="Data Preprocessing",
#     script="src/data_preprocessing.py",
#     working_directory=".",
#     arguments={
#         "--input_path": "outputs/cleaned_data.csv",
#         "--output_train": "outputs/X_train.pkl",
#         "--output_test": "outputs/X_test.pkl"
#     },
#     task_type=Task.TaskTypes.data_processing
# )

# # Step 3: Model Training
# pipeline.add_step(
#     name="model_training",
#     parents=["data_preprocessing"],
#     base_task_project="AI for Diabetes Prediction",
#     base_task_name="Model Training",
#     script="src/model_training.py",
#     working_directory=".",
#     arguments={
#         "--train_data": "outputs/X_train.pkl",
#         "--model_output": "outputs/trained_model.pkl"
#     },
#     task_type=Task.TaskTypes.training
# )

# # Step 4: Model Evaluation
# pipeline.add_step(
#     name="model_evaluation",
#     parents=["model_training"],
#     base_task_project="AI for Diabetes Prediction",
#     base_task_name="Model Evaluation",
#     script="src/model_evaluation.py",
#     working_directory=".",
#     arguments={
#         "--model_path": "outputs/trained_model.pkl",
#         "--test_data": "outputs/X_test.pkl",
#         "--metrics_output": "outputs/metrics.json"
#     },
#     task_type=Task.TaskTypes.testing
# )

# # Step 5: Model Selection
# pipeline.add_step(
#     name="model_selection",
#     parents=["model_evaluation"],
#     base_task_project="AI for Diabetes Prediction",
#     base_task_name="Model Selection",
#     script="src/model_selection.py",
#     working_directory=".",
#     arguments={
#         "--metrics_path": "outputs/metrics.json",
#         "--selected_model_path": "outputs/best_model.pkl"
#     },
#     task_type=Task.TaskTypes.inference
# )

# # Start pipeline execution
# pipeline.start()
# pipeline.wait()
# print("ClearML pipeline execution completed.")

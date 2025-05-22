from clearml import Task
import os
import shutil

# Step 1: Find the latest model_selection task
selection_tasks = Task.get_tasks(project_name="AI for Diabetes Prediction", task_name="model_selection")
latest_selection_task = selection_tasks[-1]

# Step 2: Download and parse the best model name
artifact_path = latest_selection_task.artifacts["best_model_name"].get_local_copy()

with open(artifact_path, "r") as f:
    line = f.read().strip()
    best_model = line.split(":")[-1].strip()  # e.g., "Random Forest"

print(f"Best model chosen: {best_model}")

# Step 3: Map to training task name and artifact name
model_to_task_name = {
    "Random Forest": "train_rf",
    "SVM": "train_svm",
    "Logistic Regression": "train_logreg"
}
model_to_artifact_key = {
    "Random Forest": "trained_model_random_forest",
    "SVM": "trained_model_svm",
    "Logistic Regression": "trained_model_logistic_regression"
}

train_task_name = model_to_task_name[best_model]
artifact_key = model_to_artifact_key[best_model]

# Step 4: Get latest training task for this model
train_tasks = Task.get_tasks(project_name="AI for Diabetes Prediction", task_name=train_task_name)
latest_train_task = train_tasks[-1]

# Step 5: Download model
model_local_path = latest_train_task.artifacts[artifact_key].get_local_copy()

# Step 6: Save to unified deploy path
os.makedirs("deployed_models", exist_ok=True)
shutil.copy(model_local_path, "deployed_models/best_model.joblib")
print("Best model copied to deployed_models/best_model.joblib")

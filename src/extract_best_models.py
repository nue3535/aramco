import argparse
import os
import shutil
from clearml import Task, StorageManager

def main(args):
    task = Task.current_task()

    try:
        model_selection_task = Task.get_task(task_id=args["model_selection_id"])

        # Download the artifact named "best_model"
        model_path = model_selection_task.artifacts["best_model"].get_local_copy()

        # Copy it to a deployable path
        os.makedirs(os.path.dirname(args["output_model_path"]), exist_ok=True)
        shutil.copy(model_path, args["output_model_path"])
        print(f"Best model saved to: {args['output_model_path']}")

    except Exception as e:
        # Fail-safe for early pipeline registration
        print("Skipped extract best model logic during registration. Reason:", str(e))
        pass

if __name__ == "__main__":
    task = Task.init(project_name="AI for Diabetes Prediction", task_name="Template - Extract Best Model")
    task.execute_remotely()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_selection_id", type=str, default="model_selection_dummy_id")
    parser.add_argument("--output_model_path", type=str, default="deployed_models/best_model.joblib")
    args = vars(parser.parse_args())

    main(args)

from src.data_loader import load_data, clean_data, preprocess_data
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_svm,
    evaluate_model,
    save_model
)
from src.evaluate import (
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)

def main():
    # Step 1: Load and preprocess data
    print("Loading data...")
    df = load_data("data/diabetes.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Step 2: Train all models
    print("\nTraining models...")
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "SVM": train_svm(X_train, y_train)
    }

    # Step 3: Evaluate all models
    print("\nEvaluating models...")
    results = {}
    for name, model in models.items():
        print(f"\n🔍 {name}")
        acc, y_pred = evaluate_model(model, X_test, y_test)
        results[name] = {
            "accuracy": acc,
            "model": model,
            "y_pred": y_pred
        }
        print(f"Accuracy: {acc:.4f}")
        print_classification_report(y_test, y_pred)
        plot_confusion_matrix(y_test, y_pred, model_name=name)

        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
            plot_roc_curve(y_test, y_scores, model_name=name)

    # Step 4: Select and save best model
    best_model_name = max(results, key=lambda x: results[x]["accuracy"])
    best_model = results[best_model_name]["model"]
    print(f"\n🏆 Best model: {best_model_name} with Accuracy = {results[best_model_name]['accuracy']:.4f}")
    save_model(best_model, path=f"outputs/models/best_model_{best_model_name.replace(' ', '_').lower()}.pkl")

    # Feature Importance Table - Only use for tree-based models like Random Forest
    if hasattr(best_model, "feature_importances_"):
        feature_names = df.drop("Outcome", axis=1).columns
        plot_feature_importance(best_model, feature_names)

if __name__ == "__main__":
    main()
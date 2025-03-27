import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay
)

def print_classification_report(y_true, y_pred):
    """
    Print the classification report.
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Auto-generate save path if not provided
    if not save_path:
        save_path = f"outputs/reports/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Confusion matrix saved at: {save_path}")
    plt.close()



def plot_roc_curve(y_true, y_scores, model_name="Model", save_path=None):
    """
    Plot and save ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")

    # Auto-generate save path if not provided
    if not save_path:
        save_path = f"outputs/reports/roc_curve_{model_name.replace(' ', '_').lower()}.png"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"ROC curve saved at: {save_path}")
    plt.close()


from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the model and returns a dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    
    # Check if model supports predict_proba for ROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred # Fallback if probability not available
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"--- Evaluation for {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc_score': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def plot_confusion_matrix(cm, model_name="Model", save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_test, y_prob, model_name="Model", save_path=None):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {model_name}')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def compare_models(metrics_list, save_dir=None):
    """
    Plots comparison bar charts for multiple models.
    """
    df = pd.DataFrame(metrics_list)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y=metric, data=df)
        plt.title(f'Model Comparison: {metric.capitalize()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_dir:
            path = f"{save_dir}/comparison_{metric}.png"
            plt.savefig(path)
            print(f"Plot saved to {path}")
            plt.close()
        else:
            plt.show()

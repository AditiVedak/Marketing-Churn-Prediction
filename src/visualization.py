import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_null_values(df):
    count_null_df = pd.DataFrame({
        'columns': df.columns,
        'percentage_null_values': round(df.isna().sum() * 100 / len(df), 2)
    })
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x='columns', y='percentage_null_values', data=count_null_df)
    ax.bar_label(ax.containers[0])
    plt.title('Percentage of null values with respect to Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, model_name="Model"):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importance for {model_name}')
        plt.barh(range(len(indices)), importances[indices], color='red', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()
    else:
        print(f"{model_name} does not provide feature importances.")

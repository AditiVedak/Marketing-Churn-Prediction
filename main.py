import os
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import clean_data, preprocess_features, split_and_resample
from src.visualization import plot_null_values, plot_correlation
from src.models import ModelFactory
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, compare_models

def main():
    # 1. Load Data
    DATA_PATH = os.path.join("dataset", "bank-full.csv")
    df = load_data(DATA_PATH)
    
    if df is None:
        return

    # 2. EDA Snippets (Optional: enable based on flags or interactive)
    print("Initial Shape:", df.shape)
    # plot_null_values(df)

    # 3. Preprocessing
    print("Cleaning and Preprocessing...")
    df = clean_data(df)
    df = preprocess_features(df)
    
    # 4. Split and Resample
    print("Splitting and Resampling...")
    X_train, X_test, y_train, y_test = split_and_resample(df, target_col='y')
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # 5. Model Training and Evaluation
    models_to_train = [
        ('LogisticRegression', {'C': [0.1, 1, 10]}),
        ('DecisionTree', {'max_depth': [4, 6, 8]}),
        ('RandomForest', {'n_estimators': [50, 100], 'max_depth': [4, 6]}),
        ('XGBoost', {'n_estimators': [50, 100], 'max_depth': [4, 6]})
    ]
    
    results = []
    
    for model_name, param_grid in models_to_train:
        print(f"\nTraining {model_name}...")
        
        # Get base model
        base_model = ModelFactory.get_model(model_name)
        
        # Train with CV
        grid = ModelFactory.train_with_cv(base_model, X_train, y_train, param_grid)
        best_model = grid.best_estimator_
        
        print(f"Best Params for {model_name}: {grid.best_params_}")
        
        # Evaluate
        metrics = evaluate_model(best_model, X_test, y_test, model_name)
        results.append(metrics)
        
        # Plots
        # plot_confusion_matrix(metrics['confusion_matrix'], model_name)
        # plot_roc_curve(y_test, metrics['y_prob'], model_name)

    # 6. Comparison
    print("\n--- Model Comparison ---")
    results_df = pd.DataFrame(results)
    print(results_df[['model', 'accuracy', 'f1_score', 'roc_auc_score']])
    
    # compare_models(results)

if __name__ == "__main__":
    main()

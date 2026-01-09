import os
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import clean_data, preprocess_features, split_and_resample
from src.visualization import plot_null_values, plot_correlation, plot_feature_importance
from src.models import ModelFactory
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, compare_models

def main():
    # Setup plots directory
    PLOTS_DIR = "plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load Data
    DATA_PATH = os.path.join("dataset", "bank-full.csv")
    df = load_data(DATA_PATH)
    
    if df is None:
        return

    # 2. EDA Snippets
    print("Initial Shape:", df.shape)
    plot_null_values(df, save_path=os.path.join(PLOTS_DIR, "null_values.png"))
    
    # Simple preprocessing for correlation (just numbers)
    # Note: Correlation needs numerical data, let's do it after encoding or on a subset
    # plot_correlation(df.select_dtypes(include='number'), save_path=os.path.join(PLOTS_DIR, "correlation.png"))

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
    feature_names = df.drop(columns=['y']).columns

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
        plot_confusion_matrix(metrics['confusion_matrix'], model_name, 
                              save_path=os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png"))
        plot_roc_curve(y_test, metrics['y_prob'], model_name, 
                       save_path=os.path.join(PLOTS_DIR, f"roc_curve_{model_name}.png"))
        
        # Feature Importance (if applicable)
        if model_name in ['DecisionTree', 'RandomForest', 'XGBoost']:
            plot_feature_importance(best_model, feature_names, model_name,
                                    save_path=os.path.join(PLOTS_DIR, f"feature_importance_{model_name}.png"))

    # 6. Comparison
    print("\n--- Model Comparison ---")
    results_df = pd.DataFrame(results)
    print(results_df[['model', 'accuracy', 'f1_score', 'roc_auc_score']])
    
    compare_models(results, save_dir=PLOTS_DIR)

if __name__ == "__main__":
    main()

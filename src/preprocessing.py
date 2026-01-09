import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def clean_data(df):
    """
    Performs initial data cleaning: handling duplicates, unknowns, and nulls.
    """
    # Replace unknown with NaN
    df = df.replace('unknown', np.nan)
    
    # Drop columns with excessive nulls (e.g., poutcome based on analysis)
    if 'poutcome' in df.columns:
        df.drop(columns='poutcome', inplace=True)
    
    # Fill NaN with mode
    for col in ['contact', 'education', 'job']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Handle outliers using IQR for specific columns
    outlier_var = ['age', 'balance', 'duration', 'campaign']
    for i in outlier_var:
        if i in df.columns:
            Q1 = df[i].quantile(0.25)
            Q3 = df[i].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            
            df.loc[(df[i] > upper_limit), i] = upper_limit
            df.loc[(df[i] < lower_limit), i] = lower_limit
            
    return df

def preprocess_features(df):
    """
    Handles encoding of categorical variables.
    """
    # Label Encoding mappings
    mappings = {
        'marital': {'single': 0, 'married': 1, 'divorced': 2},
        'education': {'secondary': 0, 'tertiary': 1, 'primary': 2},
        'default': {'yes': 1, 'no': 0},
        'housing': {'yes': 1, 'no': 0},
        'loan': {'yes': 1, 'no': 0},
        'contact': {'cellular': 1, 'telephone': 0},
        'y': {'yes': 1, 'no': 0}
    }
    
    for col, mapping in mappings.items():
        if col in df.columns:
            # Handle values not in mapping just in case, though cleaning should handle it
            df[col] = df[col].map(mapping)
            
    # One-hot encoding for job and month
    cols_to_dummy = ['job', 'month']
    existing_cols = [c for c in cols_to_dummy if c in df.columns]
    if existing_cols:
        df = pd.get_dummies(df, columns=existing_cols, prefix=existing_cols, drop_first=True)
        
    return df

def split_and_resample(df, target_col='y', test_size=0.2, random_state=42):
    """
    Splits the data, handles class imbalance with SMOTE, and scales features.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle Class Imbalance
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=random_state
    )
    
    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

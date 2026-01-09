import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Loads the dataset from the specified filepath.
    """
    try:
        df = pd.read_csv(filepath, sep=';')
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

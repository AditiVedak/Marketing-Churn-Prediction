# Market Campaign Churn Prediction

This project analyzes a bank's marketing campaign data to predict customer churn (subscription to a term deposit).

## Project Structure
```
.
├── dataset/
│   └── bank-full.csv       # The dataset
├── src/
│   ├── data_loader.py      # Data loading logic
│   ├── preprocessing.py    # Data cleaning, encoding, SMOTE
│   ├── visualization.py    # Plotting functions
│   ├── models.py           # Model definitions (LR, DT, RF, XGB, etc.)
│   └── evaluation.py       # Metrics and evaluation plots
├── main.py                 # Main execution script
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Setup via Terminal

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/AditiVedak/Marketing-Churn-Prediction.git
    cd Marketing-Churn-Prediction
    ```

2.  **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script to process data, train models, and view results:

```bash
python main.py
```

The script will:
1. Load the data from `dataset/bank-full.csv`.
2. Clean and preprocess the data (handle missing values, encode categoricals).
3. Handle class imbalance using SMOTE.
4. Train multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost) using Cross-Validation.
5. output performance metrics (Accuracy, F1-Score, ROC-AUC) for the best models.

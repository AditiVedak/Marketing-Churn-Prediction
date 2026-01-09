# Bank Marketing Campaign Effectiveness Prediction

## Project Overview

This project predicts whether a customer will subscribe to a **term deposit** based on data from a Portuguese bank’s marketing campaigns.
The original Jupyter Notebook was **fully refactored into a modular, production-ready Python project**, verified locally, and deployed to GitHub.

The solution demonstrates **end-to-end machine learning engineering**, from data preprocessing and modeling to evaluation.

---

## Business Problem

Bank marketing campaigns are costly and have a **low conversion rate (~11.7%)**.
The objective is to **identify high-probability customers** in advance, helping banks:

* Improve campaign efficiency
* Reduce unnecessary calls
* Increase subscription rates

---

## Dataset

* **Source:** Portuguese Bank Marketing Dataset
* **Records:** 45,211 customers
* **Features:** 17 (demographic, financial, and campaign-related)
* **Target:** `y` → Term deposit subscription (Yes / No)

### Key Characteristics

* **Class imbalance:**

  * Subscribed: **11.7%**
  * Not Subscribed: **88.3%**
* Categorical and numerical features
* Missing / “unknown” values handled appropriately

---

## Methodology

### 1. Data Preprocessing

* Removed features with >50% missing values
* Replaced missing categorical values with mode
* Outlier treatment using **Interquartile Range (IQR)**
* Encoding:

  * Label Encoding (low-cardinality features)
  * One-Hot Encoding (job, month)
* Feature scaling using **MinMaxScaler**
* Class imbalance handled using **SMOTE**

---

### 2. Model Training

The following models were trained and evaluated using **train-test split + cross-validation**:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Hyperparameter tuning was applied where applicable.

---

## Model Performance

### Final Evaluation Results

| Model               | Accuracy   | F1 Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.9249     | 0.9275     | 0.9250     |
| Decision Tree       | 0.8806     | 0.8878     | 0.8806     |
| Random Forest       | 0.9081     | 0.9162     | 0.9081     |
| **XGBoost**         | **0.9320** | **0.9349** | **0.9320** |

**XGBoost achieved the best overall performance across all metrics**.

---

## Best Model: XGBoost

The **XGBoost classifier** was selected as the final model due to:

* **Accuracy:** 0.93
* **Precision:** 0.93
* **Recall:** 0.93
* **F1 Score:** 0.93
* **ROC-AUC:** 0.93

This indicates the model can **reliably distinguish between subscribing and non-subscribing customers**, even with imbalanced data.

---

## Key Insights from Data Analysis

* Customers aged **30–36** are most likely to subscribe
* Clients with **no loans** are more likely to subscribe
* Customers contacted via **cellular** show higher conversion
* Longer call duration increases subscription probability
* Customers contacted more than **3 times never subscribed**
* Customers with **management roles and tertiary education** show higher conversion

---

## Model Explainability

* **SHAP** can be used to explain XGBoost predictions (library included in requirements)
* Features like **housing loan status and month of contact** typically have strong influence
* Lower values of most features positively impacted subscription probability

---

## Project Structure

```
├── dataset/
│   └── bank-full.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Activate Virtual Environment

**Windows**

```bash
.\venv\Scripts\activate
```

### 2. Run the Project

```bash
python main.py
```

---

## Verification Output (Sample)

```
Data loaded successfully from dataset\bank-full.csv. Shape: (45211, 17)
Train Shape: (54902, 59), Test Shape: (13726, 59)

--- Model Comparison ---
XGBoost Accuracy: 0.9320
F1 Score: 0.9349
ROC-AUC: 0.9320
```
---
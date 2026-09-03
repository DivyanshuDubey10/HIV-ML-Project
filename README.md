# HIV Infection Classification System using Machine Learning

An end-to-end machine learning project that predicts HIV infection status using clinical and demographic features from the ACTG175 dataset.

## Project Overview

This project demonstrates a complete machine learning workflow:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Train/test splitting
- Logistic Regression baseline
- Random Forest classification
- 5-fold Cross-Validation
- Hyperparameter tuning using GridSearchCV
- Model evaluation
- Model persistence using Joblib
- Interactive Streamlit web application

## Dataset

The dataset contains 2,139 records and 22 input features.

The target variable is:

- `0` — Not infected
- `1` — Infected

The model uses clinical, demographic, treatment, and laboratory-related features.

## Machine Learning Model

Two models were evaluated:

1. Logistic Regression
2. Random Forest Classifier

Random Forest performed better and was further optimized using `GridSearchCV`.

### Final Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | 89.72% |
| 5-Fold Cross-Validation Accuracy | 89.07% |
| ROC-AUC | 0.92447 |
| Class 1 Precision | 82% |
| Class 1 Recall | 74% |
| Class 1 F1-Score | 78% |

### Best Random Forest Parameters

```text
n_estimators = 100
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
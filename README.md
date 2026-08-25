# Customer Churn Analysis & Prediction

An end-to-end Python project that analyzes telecom customer churn and builds a Random Forest classification model to predict customers at risk of leaving.

## Project Overview

Customer churn can directly affect recurring revenue and customer lifetime value. This project explores customer demographics, services, contracts, tenure, and billing behavior to identify churn patterns and build a predictive model.

### Objectives

- Perform exploratory data analysis (EDA)
- Identify customer segments associated with churn
- Clean and preprocess numerical and categorical data
- Build a Random Forest classification model
- Evaluate the model using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix
- Identify important features for churn prediction
- Translate analytical findings into practical customer-retention considerations

## Dataset

**Telco Customer Churn** dataset.

The notebook loads the public dataset from IBM's archived sample repository at runtime, so the raw dataset is intentionally not included in this repository.

- Dataset size: approximately 7,000 customers
- Target: `Churn`
- Key variables include tenure, contract type, internet service, payment method, monthly charges, and total charges.

Dataset reference:
https://github.com/IBM/telco-customer-churn-on-icp4d

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Machine Learning Workflow

1. Load the Telco customer dataset
2. Inspect data types, missing values, and duplicates
3. Convert `TotalCharges` to numeric
4. Explore churn distribution and major customer segments
5. Separate predictors and target
6. Apply preprocessing using a Scikit-learn pipeline
7. Train/test split with stratification
8. Train a Random Forest classifier
9. Evaluate classification performance
10. Analyze feature importance
11. Translate findings into business-oriented retention hypotheses

## How to Run

### Option 1 — Google Colab

Open `customer_churn_analysis_prediction.ipynb` in Google Colab and run the cells sequentially.

### Option 2 — Local Python Environment

```bash
pip install -r requirements.txt
jupyter notebook customer_churn_analysis_prediction.ipynb
```

The notebook downloads the dataset from the public IBM GitHub source when executed.

## Repository Structure

```text
customer_churn_analysis_prediction/
│
├── customer_churn_analysis_prediction.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── README.md
```

## Portfolio Value

This project demonstrates practical skills in:

- Data cleaning
- Exploratory data analysis
- Data visualization
- Feature preprocessing
- Supervised machine learning
- Classification model evaluation
- Feature importance analysis
- Business interpretation of analytical results

## Attribution

The project topic and dataset workflow were informed by the **Customer Churn Analysis Prediction** tutorial from GeeksforGeeks and the IBM Telco Customer Churn sample dataset. The notebook in this repository is an independently structured implementation with its own preprocessing pipeline, evaluation workflow, and business interpretation.

## Future Enhancements

- Compare Logistic Regression, Decision Tree, Random Forest and Gradient Boosting
- Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
- Threshold optimization for recall-focused retention campaigns
- SHAP model explainability
- Interactive Streamlit dashboard
- Customer-level churn-risk scoring

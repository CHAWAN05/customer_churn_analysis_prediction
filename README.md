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
## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
customer_churn_analysis_prediction/
│
├── data/
├── Customer_Churn_Analysis_Prediction.ipynb
├── customer_churn_analysis_prediction.py
├── churn_distribution.png
├── confusion_matrix.png
├── feature_importance.png
├── requirements.txt
└── README.md
```

## Project Workflow

1. Load and inspect the Telco Customer Churn dataset
2. Analyze customer churn distribution
3. Handle missing and incorrect values
4. Encode categorical variables
5. Split data into training and testing sets
6. Apply feature scaling using StandardScaler
7. Train a Random Forest Classifier
8. Evaluate model performance using accuracy and a confusion matrix

## Model Results & Key Findings

The Random Forest classification model was evaluated on a held-out test set of **1,409 customers**.

### Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 78.50% |
| Precision | 0.62 |
| Recall | 0.50 |
| F1-Score | 0.55 |
| ROC-AUC | 0.825 |

The overall customer churn rate in the dataset was **26.54%**.

### Confusion Matrix

The model produced the following results on the test set:

- **927** customers correctly classified as No Churn
- **108** customers incorrectly classified as Churn
- **200** churn customers incorrectly classified as No Churn
- **174** customers correctly identified as Churn

The model correctly identified **174 of 374 actual churn customers** in the test set.

## Visual Analysis

### Customer Churn Distribution

The dataset shows an overall customer churn rate of **26.54%**, indicating that approximately one in four customers has churned.

![Customer Churn Distribution](churn_distribution.png)

---

### Model Evaluation — Confusion Matrix

The confusion matrix shows the model's classification performance on the held-out test set.

![Customer Churn Confusion Matrix](confusion_matrix.png)

---

### Feature Importance

The Random Forest model identified the following variables among the most influential predictors of customer churn:

1. Total Charges
2. Tenure
3. Monthly Charges
4. Month-to-month Contract
5. Online Security
6. Two-year Contract
7. Tech Support
8. Electronic Check Payment Method

![Top Features Influencing Churn Prediction](feature_importance.png)

---

## Conclusion

The analysis demonstrates that customer tenure, billing behavior, contract type, and selected service features play an important role in predicting customer churn.

The Random Forest model achieved 78.50% accuracy and a 0.825 ROC-AUC on the evaluated test set.

These insights can support targeted customer-retention strategies, particularly for customers with shorter tenure, higher monthly charges, and month-to-month contracts.

## Business Insights

The analysis suggests several areas that could be investigated for customer-retention strategies:

- Prioritize customers on month-to-month contracts for retention initiatives.
- Monitor customers with relatively high monthly charges.
- Pay attention to early-tenure customers during onboarding and service-quality initiatives.
- Investigate service combinations associated with higher predicted churn risk.
- Use predicted churn probabilities to prioritize limited retention resources.

These are analytical hypotheses rather than causal conclusions and should be validated through further business testing.

## Dataset

**Telco Customer Churn** dataset.

The project uses the IBM Telco Customer Churn sample dataset. A copy of the dataset is included in the repository for reproducibility.

- Dataset size: approximately 7,000 customers
- Target: `Churn`
- Key variables include tenure, contract type, internet service, payment method, monthly charges, and total charges.

Dataset reference:
https://github.com/IBM/telco-customer-churn-on-icp4d

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
├── data/
├── Customer_Churn_Analysis_Prediction.ipynb
├── customer_churn_analysis_prediction.py
├── churn_distribution.png
├── confusion_matrix.png
├── feature_importance.png
├── requirements.txt
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

This project was independently implemented using the IBM Telco Customer Churn sample dataset.

The project topic and general workflow were informed by publicly available customer churn analysis resources, including a GeeksforGeeks tutorial. The implementation in this repository includes its own preprocessing pipeline, model evaluation, visualizations, feature analysis, and business interpretation.

## Future Enhancements

- Compare Logistic Regression, Decision Tree, Random Forest and Gradient Boosting
- Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
- Threshold optimization for recall-focused retention campaigns
- SHAP model explainability
- Interactive Streamlit dashboard
- Customer-level churn-risk scoring

# Customer Churn Analysis & Prediction
# Converted from the final Google Colab/Jupyter notebook.
# Dataset: Telco Customer Churn

# # Customer Churn Analysis & Prediction
#
# ## Project Overview
#
# Customer churn occurs when a customer stops using a company's service, which can lead to revenue loss and affect business growth.
#
# This project analyzes the **Telco Customer Churn dataset** to understand churn patterns and build a **Random Forest classification model** to predict whether a customer is likely to churn.
#
# ### Objectives
#
# - Understand the customer churn dataset
# - Perform data cleaning and preprocessing
# - Analyze churn distribution
# - Convert categorical variables into numerical features
# - Train a Random Forest classification model
# - Evaluate model performance using multiple metrics
# - Identify important features associated with model predictions
# - Derive practical business insights

# ## 1. Import Libraries
#
# The required Python libraries are imported for data manipulation, visualization, preprocessing, model training, and evaluation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ## 2. Load Dataset
#
# The Telco Customer Churn dataset is uploaded and loaded into a Pandas DataFrame for analysis. The upload method uses the actual filename selected by the user, so the notebook is not dependent on a fixed local filename.

from google.colab import files

uploaded = files.upload()
file_name = next(iter(uploaded))

dataset = pd.read_csv("data/Telco-Customer-Churn.csv")

dataset.head()

# ## 3. Understand the Dataset
#
# Before building the model, the dataset is inspected to understand its size, structure, data types, missing values, duplicate records, and numerical statistics.

# ### Dataset Shape

print("Dataset shape:", dataset.shape)

# ### Dataset Information

dataset.info()

# ### Missing Values

dataset.isnull().sum()

# ### Duplicate Records

print("Duplicate records:", dataset.duplicated().sum())

# ### Statistical Summary

dataset.describe()

# ## 4. Churn Analysis
#
# The target variable, **Churn**, is analyzed to understand how many customers stayed with the service and how many left.

# ### Churn Counts

churn_counts = dataset["Churn"].value_counts()
churn_counts

# ### Churn Percentage

churn_percentage = dataset["Churn"].value_counts(normalize=True) * 100
churn_percentage.round(2)

# ### Churn Distribution

plt.figure(figsize=(6, 4))
sns.countplot(data=dataset, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

# ### Churn Analysis Summary
#
# Approximately **26.5%** of customers have churned, while **73.5%** have remained with the company. Churn is therefore the minority class, which is important when interpreting classification metrics, especially recall for churned customers.

# ## 5. Data Preprocessing
#
# The dataset is prepared for machine learning by converting `TotalCharges` to numeric format, handling missing values, removing the customer identifier, encoding the target variable, and converting categorical features into numerical dummy variables.

# ### Convert Total Charges to Numeric

dataset["TotalCharges"] = pd.to_numeric(
    dataset["TotalCharges"],
    errors="coerce"
)

print("Missing TotalCharges after conversion:", dataset["TotalCharges"].isnull().sum())

# ### Handle Missing Values
#
# The numeric conversion creates **11 missing values** in `TotalCharges`. These values are replaced with 0 because they correspond to customers with very short or zero tenure in the dataset.

dataset["TotalCharges"] = dataset["TotalCharges"].fillna(0)

print("Remaining missing TotalCharges:", dataset["TotalCharges"].isnull().sum())

# ### Remove Duplicate Records
#
# Duplicate rows are removed if any are present. For this dataset, the duplicate count is expected to be zero.

duplicate_count = dataset.duplicated().sum()
print("Duplicate records before removal:", duplicate_count)

if duplicate_count > 0:
    dataset = dataset.drop_duplicates()

print("Dataset shape after duplicate handling:", dataset.shape)

# ### Remove Customer Identifier
#
# `customerID` is removed because it is a unique identifier and does not provide meaningful predictive information for churn.

dataset = dataset.drop("customerID", axis=1)

dataset.head()

# ### Encode Churn Target
#
# The target variable is converted into binary numerical values:
#
# - `No` → 0
# - `Yes` → 1

dataset["Churn"] = dataset["Churn"].map({"No": 0, "Yes": 1})

dataset["Churn"].value_counts()

# ### Encode Categorical Features
#
# The remaining categorical variables are converted into numerical dummy variables using one-hot encoding. `drop_first=True` reduces redundant dummy columns.

categorical_cols = dataset.select_dtypes(include="object").columns

dataset = pd.get_dummies(
    dataset,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

print("Preprocessed dataset shape:", dataset.shape)
dataset.head()

# ## 6. Prepare Data for Modeling
#
# The preprocessed data is separated into predictor variables (`X`) and the target variable (`y`), followed by a stratified train-test split.
#
# **Note:** Random Forest is a tree-based algorithm and does not require feature scaling, so StandardScaler is not necessary for this model.

# ### Separate Features and Target

X = dataset.drop("Churn", axis=1)
y = dataset["Churn"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ### Train-Test Split
#
# 80% of the data is used for training and 20% for testing. Stratification preserves the churn/non-churn class proportion in both sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# ## 7. Model Training
#
# A **Random Forest Classifier** is trained to predict customer churn. The model combines multiple decision trees to produce a more robust classification model.

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Random Forest model trained successfully.")

# ### Generate Predictions

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully.")

# ## 8. Model Evaluation
#
# The model is evaluated using accuracy, precision, recall, F1-score, ROC-AUC, a classification report, and a confusion matrix. Multiple metrics are used because churn is the minority class.

# ### Performance Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy : {accuracy:.2%}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1-Score : {f1:.2f}")
print(f"ROC-AUC  : {roc_auc:.3f}")

# ### Classification Report

print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
))

# ### Confusion Matrix
#
# The confusion matrix shows the number of correctly and incorrectly classified customers in the non-churn and churn classes.

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("Confusion Matrix - Customer Churn Prediction")
plt.show()

# ### Performance Summary

performance = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Score": [accuracy, precision, recall, f1, roc_auc]
})

performance["Score (%)"] = performance["Score"] * 100
performance

# ## 9. Feature Importance
#
# Random Forest provides feature importance scores that indicate which variables contributed most to the model's predictions. These scores describe model behavior and should not be interpreted as proof of causation.

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("Top 10 important features:")
display(feature_importance.head(10).to_frame("Importance"))

plt.figure(figsize=(8, 6))

feature_importance.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Features Influencing Churn Prediction")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.show()

# ## 10. Key Findings
#
# - The dataset contains **7,043 customer records**.
# - Approximately **26.5% of customers have churned**.
# - The Random Forest model achieves approximately **78.5% accuracy** on the held-out test set.
# - The churn class has lower recall than the non-churn class, indicating that some customers who actually churn are not identified by the baseline model.
# - Feature importance analysis indicates that variables such as **TotalCharges, tenure, MonthlyCharges, payment method, internet service, and contract-related features** contribute strongly to the model's predictions.
# - These findings provide a baseline for further churn-prediction improvements.

# ## 11. Business Insights
#
# The model can support customer-retention activities by helping businesses identify customers who may be at higher risk of leaving.
#
# Potential applications include:
#
# - Prioritize customers on **month-to-month contracts** for retention initiatives.
# - Monitor customers with **higher monthly charges** and shorter tenure.
# - Pay attention to **early-tenure customers** during onboarding and service-quality initiatives.
# - Review service and payment-method combinations associated with higher predicted churn risk.
# - Use predicted churn probabilities to prioritize limited retention resources.
#
# These are **analytical findings rather than causal conclusions** and should be validated through further business testing.

# ## 12. Conclusion
#
# The project analyzed customer churn in the Telco Customer Churn dataset and developed a Random Forest classification model to predict churn.
#
# The model provides a useful baseline, achieving approximately **78.5% accuracy** and a **0.825 ROC-AUC** on the held-out test set. The evaluation also highlights the importance of recall when the business objective is to identify customers who may leave.
#
# The analysis demonstrates how data preprocessing, exploratory analysis, machine learning, model evaluation, and feature-importance analysis can be combined to support customer-retention decision making.

# ## 13. Future Enhancements
#
# - Compare Random Forest with Logistic Regression, Decision Tree, Gradient Boosting, and other classifiers.
# - Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
# - Optimize the classification threshold to improve recall for churned customers.
# - Apply cross-validation for more robust model evaluation.
# - Use SHAP or other explainability techniques for deeper model interpretation.
# - Develop an interactive dashboard for customer-level churn-risk analysis.

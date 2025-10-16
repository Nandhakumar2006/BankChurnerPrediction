#### 🧠 Bank Customer Churn Prediction using Decision Tree

This project aims to predict customer churn in a bank using a Decision Tree Classifier. The model identifies customers who are likely to leave the bank, enabling proactive retention strategies.

#### 📋 Project Overview

This notebook demonstrates the end-to-end machine learning pipeline for churn prediction:

Data loading and preprocessing

Handling missing values and duplicates

Exploratory Data Analysis (EDA) with correlation heatmap

Feature engineering and encoding using ColumnTransformer

Class imbalance handling using SMOTE (Synthetic Minority Oversampling Technique)

Model training using Decision Tree Classifier within an imbalanced pipeline

Hyperparameter tuning with GridSearchCV

Model evaluation using various performance metrics

#### 🧰 Technologies and Libraries Used

Python

Pandas, NumPy — Data manipulation and preprocessing

Matplotlib, Seaborn — Visualization and EDA

Scikit-learn — Model building, evaluation, and pipeline management

Imbalanced-learn (imblearn) — SMOTE for handling class imbalance

#### ⚙️ Workflow
1. Data Preparation

Dataset: Churn_Modelling.csv

Target variable: Exited (1 = Churned, 0 = Stayed)

Dropped columns: RowNumber, CustomerId, Surname

#### 2. Feature Engineering

Categorical Encoding: Applied OneHotEncoder to categorical columns

Numerical Columns: Retained as-is through the pipeline

#### 3. Imbalanced Data Handling

Applied SMOTE within the pipeline to generate synthetic samples for minority class.

#### 4. Model Building

Model used: DecisionTreeClassifier(random_state=42)

Wrapped in an ImbPipeline combining preprocessing, SMOTE, and classifier

#### 5. Hyperparameter Tuning

Performed grid search using:

param_grid = {
    "model__max_depth": [3, 5, 7, 9, None],
    "model__min_samples_split": [2, 5, 10, 20]
}


Optimized using 5-fold cross-validation and ROC-AUC as the scoring metric.

#### 6. Model Evaluation

Evaluated on the test set using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Classification Report

Also determined the optimal probability threshold for the best F1 score using precision_recall_curve.

#### 📊 Key Outputs

Best Hyperparameters: (example output from GridSearchCV)

Best params: {'model__max_depth': 5, 'model__min_samples_split': 5}
Best CV ROC-AUC: 0.8466


#### Test Set Performance:

Metric	Score
Accuracy	0.8305
Precision	0.5749
Recall	0.6413
F1-Score	0.6063
ROC-AUC	0.8466

#### Visualization:

Correlation heatmap to explore feature relationships

Decision tree structure (optional visualization via plot_tree)

#### 🧩 Insights

The Decision Tree performs reasonably well with a balanced trade-off between precision and recall.

ROC-AUC of 0.8466 indicates strong discriminative power.

Using SMOTE effectively addresses the imbalance issue and improves recall.



###### APP LINK

https://huggingface.co/spaces/nandha-01/BankChurnPredictor

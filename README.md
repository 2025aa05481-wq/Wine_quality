# Machine Learning Assignment 2 - Wine Quality Classification

## Project Overview

This project implements a comprehensive machine learning classification system using **6 different algorithms** to predict wine quality based on physicochemical properties. The system is deployed as an interactive Streamlit web application with real-time prediction capabilities and comprehensive model evaluation.

## Model Performance Results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.7406 | 0.8242 | 0.7419 | 0.7406 | 0.7409 | 0.4808 |
| Decision Tree | 0.7531 | 0.7513 | 0.7529 | 0.7531 | 0.7529 | 0.5034 |
| K-Nearest Neighbor | 0.7406 | 0.8117 | 0.7407 | 0.7406 | 0.7407 | 0.4790 |
| Naive Bayes | 0.7219 | 0.7884 | 0.7282 | 0.7219 | 0.7219 | 0.4500 |
| Random Forest | 0.8031 | 0.9020 | 0.8043 | 0.8031 | 0.8033 | 0.6062 |
| XGBoost | 0.8250 | 0.8963 | 0.8259 | 0.8250 | 0.8252 | 0.6497 |


## Observations about Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| **Logistic Regression** | Good baseline model with balanced performance across all metrics. Shows decent AUC (0.8242) indicating good discriminative ability, but moderate MCC (0.4808) suggests limited correlation strength. |
| **Decision Tree** | Slightly better accuracy than logistic regression but lower AUC (0.7513), indicating potential overfitting. The MCC (0.5034) shows moderate correlation, but the gap between accuracy and AUC suggests overfitting to training data. |
| **K-Nearest Neighbor** | Performance similar to logistic regression with identical accuracy (0.7406). Good AUC (0.8117) but lower MCC (0.4790) indicates the model struggles with consistent predictions across different thresholds. |
| **Naive Bayes** | Lowest performing model with accuracy of 0.7219. Despite having decent AUC (0.7884), the strong independence assumptions likely limit its effectiveness on this dataset with correlated features. |
| **Random Forest** | Significant performance improvement with 80.31% accuracy and highest AUC (0.9020). Excellent MCC (0.6062) indicates strong correlation and balanced performance. Ensemble approach effectively handles feature interactions. |
| **XGBoost** | Best overall performer with 82.50% accuracy and strong MCC (0.6497). Gradient boosting effectively captures complex patterns in the data. Slightly lower AUC than Random Forest but better overall balance across all metrics. |

## Web Application

https://winequality-ydf35p344smcc5v2cuavje.streamlit.app/

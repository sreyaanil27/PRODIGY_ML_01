# House Price Prediction using Linear Regression

This project was developed as part of my Machine Learning Internship at Prodigy InfoTech.

The objective of this project is to build a Linear Regression model capable of predicting house prices based on housing characteristics such as living area, number of bedrooms, bathrooms, neighborhood, garage capacity, basement area, and overall house quality.

---

# Project Overview

This project follows a complete Machine Learning workflow including:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Selection
- Multicollinearity Handling
- One-Hot Encoding
- Model Training
- Cross Validation
- Model Evaluation
- Streamlit Web Application Development

---

# Dataset

Dataset used:
House Prices - Advanced Regression Techniques (Kaggle)

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Git & GitHub

---

# Exploratory Data Analysis

EDA was performed to analyze:
- Feature distributions
- Correlations
- Outliers
- Neighborhood impact
- Housing quality impact
- Price trends

Key observations:
- OverallQual showed a strong positive correlation with SalePrice.
- Larger living areas and basement areas generally increased house prices.
- Neighborhood significantly influenced house prices.
- Outliers represented realistic premium properties and were retained.

---

# Features Used

The model was trained using the following features:

- GrLivArea
- BedroomAbvGr
- FullBath
- OverallQual
- GarageCars
- TotalBsmtSF
- YearBuilt
- LotArea
- Neighborhood

Preprocessing steps:
- One-Hot Encoding for Neighborhood
- Removal of multicollinear features

---

# Model Used

- Linear Regression

---

# Model Performance

## Cross Validation R² Scores

```text
[0.85313165, 0.82386022, 0.81104143, 0.81189815, 0.698288]
```

Average CV R² Score:

0.7996

## Train vs Test Performance

| Metric | Value |
|---|---|
| Train R² | 0.8084 |
| Test R² | 0.8324 |

## Final Evaluation Metrics

| Metric | Value |
|---|---|
| MAE | 21799.52 |
| MSE | 1285652619.08 |
| RMSE | 35856.00 |
| R² Score | 0.8324 |

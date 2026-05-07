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
[0.8531, 0.8238, 0.8110, 0.8119, 0.6982]

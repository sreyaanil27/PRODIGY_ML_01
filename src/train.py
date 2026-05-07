import pandas as pd
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# 1. Load Dataset
df = pd.read_csv("data/train.csv")


# 2. Select Features
features = [
    'GrLivArea',
    'BedroomAbvGr',
    'FullBath',
    'OverallQual',
    'GarageCars',
    'TotalBsmtSF',
    'YearBuilt',
    'Neighborhood',
    'LotArea',
    'SalePrice'
]

df = df[features]

# 3. One-Hot Encode Neighborhood
df = pd.get_dummies(
    df,
    columns=['Neighborhood'],
    drop_first=True,
    dtype=int
)


# 4. Features & Target
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']


# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Train Model
model = LinearRegression()

# Cross-validation
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='r2'
)

print("Cross-Validation R2 Scores:")
print(cv_scores)

print(f"\nAverage CV R2: {cv_scores.mean():.4f}")

# Train model
model.fit(X_train, y_train)


# 7. Predictions
y_pred = model.predict(X_test)


# 8. Train vs Test R2
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print("\nTrain R2:", round(train_r2, 4))
print("Test R2 :", round(test_r2, 4))


# 9. Evaluation Metrics

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nExtended Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")


# 10. Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(8,5))

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(0, color='red', linestyle='--')

plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.show()


# 11. Feature Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Coefficients:")
print(
    coefficients.sort_values(
        by='Coefficient',
        ascending=False
    )
)


# 12. Save Model
os.makedirs("models", exist_ok=True)

with open("models/extended_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nExtended model saved!")
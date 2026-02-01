"""
House Price Prediction in King County, USA
Linear Regression Analysis
"""

# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# Data Loading
# =========================
# Load the dataset
df = pd.read_csv("data/houseprice_data.csv")

# Preview the data
print(df.head())
print(df.tail())

# Dataset information
print(df.info())
print(df.describe())


# =========================
# Exploratory Data Analysis
# =========================
# Correlation analysis to understand relationships between features
correlation_matrix = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap="coolwarm")
plt.title("Correlation Heatmap of House Features")
plt.show()


# =========================
# Simple Linear Regression
# =========================
# Using sqft_living as a single predictor
X_simple = df[["sqft_living"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

simple_lr_model = LinearRegression()
simple_lr_model.fit(X_train, y_train)

y_pred_simple = simple_lr_model.predict(X_test)

mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print("Simple Linear Regression")
print("MSE:", mse_simple)
print("R-squared:", r2_simple)


# =========================
# Multiple Linear Regression
# =========================
# Selecting multiple predictors
features = [
    "bedrooms", "bathrooms", "sqft_living", "floors", "waterfront",
    "view", "condition", "grade", "sqft_above", "sqft_basement"
]

X_multi = df[features]

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

multi_lr_model = LinearRegression()
multi_lr_model.fit(X_train_multi, y_train_multi)

y_pred_multi = multi_lr_model.predict(X_test_multi)

mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)

print("\nMultiple Linear Regression")
print("MSE:", mse_multi)
print("R-squared:", r2_multi)


# =========================
# Extended Linear Regression
# =========================
# Adding location and time-based features
extended_features = features + ["zipcode", "yr_built", "yr_renovated"]
X_extended = df[extended_features]

X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
    X_extended, y, test_size=0.2, random_state=42
)

extended_lr_model = LinearRegression()
extended_lr_model.fit(X_train_ext, y_train_ext)

y_pred_ext = extended_lr_model.predict(X_test_ext)

# Cross-validation for robustness
cv_scores = cross_val_score(extended_lr_model, X_extended, y, cv=5)

mse_ext = mean_squared_error(y_test_ext, y_pred_ext)
r2_ext = r2_score(y_test_ext, y_pred_ext)

print("\nExtended Linear Regression")
print("Cross-validation R-squared:", cv_scores.mean())
print("MSE:", mse_ext)
print("R-squared:", r2_ext)


# =========================
# Model Visualization
# =========================
plt.figure(figsize=(10, 6))
plt.scatter(y_test_ext, y_pred_ext, alpha=0.5)
plt.plot(y_test_ext, y_test_ext, color="red")
plt.title("Actual vs Predicted House Prices (Extended Model)")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

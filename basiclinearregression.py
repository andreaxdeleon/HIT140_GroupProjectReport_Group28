import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load datasets
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")

# Merge datasets on a common column (e.g., 'ID')
merged_df = pd.merge(df1, df2, on='ID')
merged_df = pd.merge(merged_df, df3, on='ID')

# Select screen time as the feature (e.g., 'T_we' for screen time on weekends)
x = merged_df[['T_we']]  # Use 'T_we' as the screen time feature

# Select the well-being score (e.g., 'Goodme')
y = merged_df['Goodme']

# Test sizes to evaluate the model
test_sizes = [0.4, 0.5, 0.3, 0.2]

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n=====================")
    print(f"Test Size: {test_size}")
    print("=====================")

    # Intercept and Coefficient
    print("Intercept (b_0): ", model.intercept_)
    print("Coefficient (b_1): ", model.coef_[0])

    y_pred = model.predict(X_test)

    # Create a DataFrame comparing actual vs predicted values
    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print("\n--- Actual vs. Predicted ---")
    print(df_pred.head())  # Show first few rows for comparison

    # Error Metrics
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    y_max = y_test.max()
    y_min = y_test.min()
    rmse_norm = rmse / (y_max - y_min)
    r_2 = metrics.r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]
    Adj_r2 = 1 - (1 - r_2) * (n - 1) / (n - p - 1)

    print("\n--- Model Performance Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Normalized RMSE: {rmse_norm}")

    # R-squared and Adjusted R-squared
    print("\n--- R-squared and Adjusted R-squared ---")
    print(f"R-squared (R²): {r_2}")
    print(f"Adjusted R²: {Adj_r2}")

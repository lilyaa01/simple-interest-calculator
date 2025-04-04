# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor  # For predicting numeric values
from sklearn.ensemble import RandomForestRegressor  # For predicting numeric values
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Kaggle (after extracting it)
# Path should point to the correct location where you downloaded the soccer dataset (players_20.csv)
url = "path_to_your_downloaded_dataset/players_20.csv"  # Replace with the actual path to the dataset
data = pd.read_csv(url)

# Check the dataset columns and types to handle preprocessing
print(data.head())  # Inspect dataset for feature names and types

# Data Preprocessing
# Handle missing values (e.g., fill with mean for numeric columns)
data.fillna(data.mean(), inplace=True)

# Example: If we are predicting 'goals'
X = data.drop(columns=['goals'])  # Drop target column (replace with actual column)
y = data['goals']

# Convert categorical columns (if any, like 'position') to numerical format (e.g., one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Split dataset into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor Model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Random Forest Regressor Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Performance Evaluation
def evaluate_model(y_true, y_pred, model_name):
    print(f"---{model_name}---")
    print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"R-squared: {r2_score(y_true, y_pred):.2f}")
    
evaluate_model(y_test, y_pred_dt, "Decision Tree")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Hyperparameter Tuning for Random Forest (Grid Search)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Final Random Forest model with best params
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)
evaluate_model(y_test, y_pred_best_rf, "Tuned Random Forest")

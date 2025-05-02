# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Function to train and save the model
def train_and_save_model(data, target_column, model, model_filename):
    # Check if target column exists
    if target_column not in data.columns:
        raise ValueError(f"'{target_column}' not found in DataFrame columns: {data.columns.tolist()}")

    # Split the data into features and labels
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # Hyperparameter tuning (example for Logistic Regression)
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Save the best model
    joblib.dump(grid_search.best_estimator_, model_filename)

    # Evaluate the model
    y_pred = grid_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return grid_search.best_estimator_, acc, conf_matrix, report

# Load dataset
data = pd.read_csv("migraine_data.csv")

# Define the correct target column
target_column = 'Type'  # Ensure this matches the column in your dataset

# Train, evaluate, and save the model
model_filename = 'Migrain_model.joblib'
trained_model, acc, conf_matrix, report = train_and_save_model(data, target_column, LogisticRegression(), model_filename)

# Print results
print(f"Model: {trained_model}")
print(f"Accuracy: {acc:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

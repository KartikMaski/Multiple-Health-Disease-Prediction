import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('COPD_data.csv')

# Define features and target variable
features = [
    'AGE', 
    'PackHistory', 
    'MWT1', 
    'MWT2', 
    'FEV1', 
    'FVC', 
    'gender', 
    'smoking', 
    'Diabetes', 
    'hypertension'
]
target = 'copd'  # Make sure this column exists in your dataset

# Split the data into features and target
X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
model_filename = 'copd_model.sav'
scaler_filename = 'scaler.sav'
pickle.dump(model, open(model_filename, 'wb'))
pickle.dump(scaler, open(scaler_filename, 'wb'))

print("Model training complete and saved.")

import numpy as np
import pickle

# Load the trained model from the pickle file
filename = 'Kidney.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Define a function to take user input for each feature
def get_user_input():
    print("Enter the following details:")
    sg = float(input("Specific Gravity (e.g., 1.010, 1.025): "))
    htn = int(input("Hypertension (1 for Yes, 0 for No): "))
    hemo = float(input("Hemoglobin level (e.g., 13.5, 9.5): "))
    dm = int(input("Diabetes Mellitus (1 for Yes, 0 for No): "))
    al = int(input("Albumin (e.g., 0, 1, 2, 3): "))
    appet = int(input("Appetite (1 for Good, 0 for Poor): "))
    rc = float(input("Red Blood Cells count (e.g., 5.2, 3.9): "))
    pc = int(input("Pus Cell (0 for Normal, 1 for Abnormal): "))
    
    # Combine all inputs into a single array (for the model)
    user_input = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
    return user_input

# Function to predict the result based on user input
def predict_ckd():
    user_input = get_user_input()
    
    # Predict the result using the model
    prediction = loaded_model.predict(user_input)[0]
    
    if prediction == 1:
        result = "CKD (Chronic Kidney Disease)"
    else:
        result = "No CKD"
    
    print(f"\nPrediction: {result}")

# Call the prediction function
predict_ckd()

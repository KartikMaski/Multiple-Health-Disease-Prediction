import pandas as pd
import pickle

# Load the trained model and scaler
model_filename = "copd_model.sav"
scaler_filename = "scaler.sav"

# Load the model and scaler
model = pickle.load(open(model_filename, 'rb'))
scaler = pickle.load(open(scaler_filename, 'rb'))

# Define the function to get user input and make predictions
def get_user_input():
    print("Please enter the following information:")

    age = int(input("Age: "))
    pack_history = float(input("Pack History: "))
    mwt1 = float(input("MWT1: "))
    mwt2 = float(input("MWT2: "))
    fev1 = float(input("FEV1: "))
    fvc = float(input("FVC: "))
    gender = int(input("Gender (0 for Male, 1 for Female): "))
    smoking = int(input("Smoking (0 for No, 1 for Yes): "))
    diabetes = int(input("Diabetes (0 for No, 1 for Yes): "))
    hypertension = int(input("Hypertension (0 for No, 1 for Yes): "))

    # Create a DataFrame for the input
    user_data = pd.DataFrame({
        'AGE': [age],
        'PackHistory': [pack_history],
        'MWT1': [mwt1],
        'MWT2': [mwt2],
        'FEV1': [fev1],
        'FVC': [fvc],
        'gender': [gender],
        'smoking': [smoking],
        'Diabetes': [diabetes],
        'hypertension': [hypertension]
    })

    return user_data

# Make predictions based on user input
def make_prediction():
    user_data = get_user_input()

    # Preprocess the input data
    user_data_scaled = scaler.transform(user_data)

    # Make prediction
    prediction = model.predict(user_data_scaled)
    
    # Output the prediction
    if prediction[0] == 0:
        print("\nPredicted COPD Status: Yes")
    else:
        print("\nPredicted COPD Status: No")

if __name__ == "__main__":
    make_prediction()

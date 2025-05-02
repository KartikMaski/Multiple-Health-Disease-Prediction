import pandas as pd
import joblib

# Load the trained model
model_filename = 'Migrain_model.joblib'
model = joblib.load(model_filename)

# Function to get user input for the features
def get_user_input():
    print("Please enter the following information:")
    age = int(input("Enter age in years (e.g., 30): "))
    duration = float(input("Enter duration of migraine in hours (e.g., 2.5): "))
    frequency = int(input("Enter frequency of migraines per month (e.g., 3): "))
    location = int(input("Enter the location of the pain (e.g., 1 for right, 2 for left): "))
    character = int(input("Enter character of pain (e.g., 1 for throbbing, 0 for dull): "))
    intensity = int(input("Enter intensity of pain on a scale from 1 to 10 (e.g., 8): "))
    nausea = int(input("Enter 1 for nausea, 0 for no nausea: "))
    vomit = int(input("Enter 1 for vomiting, 0 for no vomiting: "))
    phonophobia = int(input("Enter 1 for phonophobia, 0 for no phonophobia: "))
    photophobia = int(input("Enter 1 for photophobia, 0 for no photophobia: "))
    visual = int(input("Enter visual disturbances (e.g., 1 for yes, 0 for no): "))
    sensory = int(input("Enter sensory disturbances (e.g., 1 for yes, 0 for no): "))
    dysphasia = int(input("Enter 1 for dysphasia, 0 for no dysphasia: "))
    dysarthria = int(input("Enter 1 for dysarthria, 0 for no dysarthria: "))
    vertigo = int(input("Enter 1 for vertigo, 0 for no vertigo: "))
    tinnitus = int(input("Enter 1 for tinnitus, 0 for no tinnitus: "))
    hypoacusis = int(input("Enter 1 for hypoacusis, 0 for no hypoacusis: "))
    diplopia = int(input("Enter 1 for diplopia, 0 for no diplopia: "))
    defect = int(input("Enter 1 for any neurological defect, 0 for none: "))
    ataxia = int(input("Enter 1 for ataxia, 0 for no ataxia: "))
    conscience = int(input("Enter 1 if consciousness is impaired, 0 if normal: "))
    paresthesia = int(input("Enter 1 for paresthesia, 0 for no paresthesia: "))
    dpf = int(input("Enter any other relevant factors (e.g., 0 or 1): "))

    return [age, duration, frequency, location, character, intensity,
            nausea, vomit, phonophobia, photophobia, visual, sensory,
            dysphasia, dysarthria, vertigo, tinnitus, hypoacusis,
            diplopia, defect, ataxia, conscience, paresthesia, dpf]

# Main function to run the tester
def main():
    features = get_user_input()
    
    # Convert features to DataFrame for prediction
    feature_names = ['Age', 'Duration', 'Frequency', 'Location', 'Character', 
                     'Intensity', 'Nausea', 'Vomit', 'Phonophobia', 'Photophobia', 
                     'Visual', 'Sensory', 'Dysphasia', 'Dysarthria', 
                     'Vertigo', 'Tinnitus', 'Hypoacusis', 'Diplopia', 
                     'Defect', 'Ataxia', 'Conscience', 'Paresthesia', 'DPF']
    
    input_data = pd.DataFrame([features], columns=feature_names)
    
    # Make prediction
    predicted_type = model.predict(input_data)[0]
    print(f"Predicted Type: {predicted_type}")

if __name__ == "__main__":
    main()

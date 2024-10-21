import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = r'D:/COLLEGE STUDIES/SEM-5/DE/Project/saved_models/Migrain_model.joblib'
model = joblib.load(model_path)

# Function to get user input for the features
def get_user_input():
    st.title("Migraine Prediction Model")
    
    age = st.number_input("Enter age in years (e.g., 30):", min_value=0, max_value=120, value=30)
    duration = st.number_input("Enter duration of migraine in hours (e.g., 2.5):", min_value=0.0, value=2.5)
    frequency = st.number_input("Enter frequency of migraines per month (e.g., 3):", min_value=0, value=3)
    location = st.selectbox("Select the location of the pain:", [1, 2], format_func=lambda x: "Right" if x == 1 else "Left")
    character = st.selectbox("Select character of pain:", [1, 0], format_func=lambda x: "Throbbing" if x == 1 else "Dull")
    intensity = st.number_input("Enter intensity of pain on a scale from 1 to 10 (e.g., 8):", min_value=1, max_value=10, value=8)
    nausea = st.selectbox("Nausea?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    vomit = st.selectbox("Vomiting?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    phonophobia = st.selectbox("Phonophobia?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    photophobia = st.selectbox("Photophobia?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    visual = st.selectbox("Visual disturbances?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    sensory = st.selectbox("Sensory disturbances?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    dysphasia = st.selectbox("Dysphasia?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    dysarthria = st.selectbox("Dysarthria?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    vertigo = st.selectbox("Vertigo?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    tinnitus = st.selectbox("Tinnitus?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    hypoacusis = st.selectbox("Hypoacusis?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    diplopia = st.selectbox("Diplopia?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    defect = st.selectbox("Any neurological defect?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    ataxia = st.selectbox("Ataxia?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    conscience = st.selectbox("Is consciousness impaired?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    paresthesia = st.selectbox("Paresthesia?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    dpf = st.number_input("Enter any other relevant factors (e.g., 0 or 1):", min_value=0, value=0)

    return [age, duration, frequency, location, character, intensity,
            nausea, vomit, phonophobia, photophobia, visual, sensory,
            dysphasia, dysarthria, vertigo, tinnitus, hypoacusis,
            diplopia, defect, ataxia, conscience, paresthesia, dpf]

# Main function to run the app
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
    if st.button("Predict"):
        predicted_type = model.predict(input_data)[0]
        st.success(f"Predicted Type: {predicted_type}")

if __name__ == "__main__":
    main()

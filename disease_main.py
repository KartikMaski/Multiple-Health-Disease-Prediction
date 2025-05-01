import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
import numpy as np

# IMPORTING MODEL
# Load models from the same directory as the script
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
COPD_model_1 = pickle.load(open('copd_model.sav', 'rb'))
COPD_model_2 = pickle.load(open('scaler.sav', 'rb'))
loaded_model = pickle.load(open('Kidney.pkl', 'rb'))
model = joblib.load('Migrain_model.joblib')




# SIDEBAR
with st.sidebar:
    # Display the image (replace 'path_to_image.png' with the path to your image)
    st.image('LOGO2.png', use_container_width=True)
    
    # Option menu for selecting prediction models
    selected = option_menu('Multiple Disease Prediction System  ',
                           ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Pulmonary Prediction', 'Kidney Prediction', 'Migraine Prediction'],
                           icons=['house', 'activity', 'heart', 'person', 'clipboard2-heart', 'bi-asterisk', 'bi-person-arms-up'],
                           default_index=0)

# HOME PAGE
if selected == 'Home':
    st.markdown("<h2 style='text-align: center;'>Welcome to Diagnosify</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your trusted partner in predicting and preventing heart disease, Parkinson's disease, and diabetes. Explore our services below.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("heart.png", use_container_width=True)  # Replace with the path to your image
        st.markdown("<h3 style='text-align: center;'>Heart Disease</h3>", unsafe_allow_html=True)
        # st.markdown("<p style='text-align: center;'>Our advanced algorithms predict the likelihood of heart disease based on various health metrics.</p>", unsafe_allow_html=True)

    with col2:
        st.image("parkinson.png", use_container_width=True)  # Replace with the path to your image
        st.markdown("<h3 style='text-align: center;'>Parkinson Disease</h3>", unsafe_allow_html=True)
        # st.markdown("<p style='text-align: center;'>Using the latest research, we provide insights into the early detection of Parkinson's disease.</p>", unsafe_allow_html=True)

    with col3:
        st.image("diabates.png", use_container_width=True)  # Replace with the path to your image
        st.markdown("<h3 style='text-align: center;'>Diabetes Disease</h3>", unsafe_allow_html=True)
        # st.markdown("<p style='text-align: center;'>Our tools help predict the onset of diabetes, allowing for early intervention and management.</p>", unsafe_allow_html=True)

   
    col4, col5, col6 = st.columns(3)

    with col4:
        st.image("kidney.png", use_container_width=True)  # Replace with the path to your image
        st.markdown("<h3 style='text-align: center;'>Kidney Disease</h3>", unsafe_allow_html=True)
        # st.markdown("<p style='text-align: center;'>Our system helps in early prediction of pulmonary disease, assisting with timely treatment.</p>", unsafe_allow_html=True)

    with col5:
        st.image("pulmonary.png", use_container_width=True)  # Replace with the path to your image
        st.markdown("<h3 style='text-align: center;'>Pulmonary Disease</h3>", unsafe_allow_html=True)
        # st.markdown("<p style='text-align: center;'>Our system helps in early prediction of pulmonary disease, assisting with timely treatment.</p>", unsafe_allow_html=True)
    
    with col6:
        st.image("migraine.png", use_container_width=True)  # Replace with the path to your image
        st.markdown("<h3 style='text-align: center;'>Migraine Disease</h3>", unsafe_allow_html=True)
        # st.markdown("<p style='text-align: center;'>Our system helps in early prediction of pulmonary disease, assisting with timely treatment.</p>", unsafe_allow_html=True)



# DIABETES PREDICTION PAGE
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    # Prediction
    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is Diabetic'
        else:
            diab_diagnosis = 'The person is NOT Diabetic'

    st.success(diab_diagnosis)

# HEART DISEASE PREDICTION PAGE
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction  ')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by fluoroscopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# PARKINSON'S DISEASE PREDICTION PAGE
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction  ")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                           Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                                                           RPDE, DFA, spread1, spread2, D2, PPE]])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

# PULMONARY DISEASE PREDICTION
if selected == 'Pulmonary Prediction':
    st.title('Pulmonary Disease Prediction')

    # Create input fields for user data collection
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age")
    with col2:
        pack_history = st.text_input("Pack History")
    with col3:
        mwt1 = st.text_input("MWT1")

    with col1:
        mwt2 = st.text_input("MWT2")
    with col2:
        fev1 = st.text_input("FEV1")
    with col3:
        fvc = st.text_input("FVC")

    with col1:
        gender = st.text_input("Gender (0 for Male, 1 for Female)")
    with col2:
        smoking = st.text_input("Smoking (0 for No, 1 for Yes)")
    with col3:
        diabetes = st.text_input("Diabetes (0 for No, 1 for Yes)")

    with col1:
        hypertension = st.text_input("Hypertension (0 for No, 1 for Yes)")

    copd_diagnosis = ""

    if st.button('COPD Test Result'):
        # Convert inputs to appropriate types
        try:
            input_data = [[
                int(age),
                float(pack_history),
                float(mwt1),
                float(mwt2),
                float(fev1),
                float(fvc),
                int(gender),
                int(smoking),
                int(diabetes),
                int(hypertension)
            ]]
            
            # Scale the input data
            user_data_scaled = COPD_model_2.transform(input_data)

            # Make the prediction using the model
            prediction = COPD_model_1.predict(user_data_scaled)

            # Display the result
            if prediction[0] == 0:
                st.success("Predicted COPD Status: Yes (At Risk)")
            else:
                st.success("Predicted COPD Status: No (Not at Risk)")
        
        except ValueError:
            st.error("Please enter valid numerical values.")

# KIDNEY DISEASE PREDICTION
if selected == "Kidney Prediction":
    st.title("Kidney Disease Prediction  ")

    # Define the function to get user input using Streamlit
    def get_user_input():
        # Create 3 columns
        col1, col2, col3 = st.columns(3)

        # Inputs distributed across the three columns
        with col1:
            sg = st.number_input("Specific Gravity (e.g., 1.010, 1.025)", min_value=1.005, max_value=1.030, value=1.020, step=0.001)
            hemo = st.number_input("Hemoglobin level (e.g., 13.5, 9.5)", min_value=5.0, max_value=17.0, value=13.5)
            rc = st.number_input("Red Blood Cells count (e.g., 5.2, 3.9)", min_value=2.0, max_value=7.0, value=5.0)

        with col2:
            htn = st.selectbox("Hypertension (1 for Yes, 0 for No)", [0, 1])
            dm = st.selectbox("Diabetes Mellitus (1 for Yes, 0 for No)", [0, 1])
            pc = st.selectbox("Pus Cell (0 for Normal, 1 for Abnormal)", [0, 1])

        with col3:
            al = st.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5])
            appet = st.selectbox("Appetite (1 for Good, 0 for Poor)", [0, 1])

        # Combine all inputs into a single array
        user_input = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
        return user_input

    # Define the prediction function
    def predict_ckd(user_input):
        prediction = loaded_model.predict(user_input)[0]
        
        if prediction == 1:
            return "CKD (Chronic Kidney Disease)"
        else:
            return "No CKD"

    # Get user input
    user_input = get_user_input()

    # Button to predict
    if st.button("Predict Kidney Disease"):
        result = predict_ckd(user_input)
        st.success(f"The prediction is: {result}")

# Migrain Prediction
def migraine_input():
    st.title("Migraine Prediction Model")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Enter age in years (e.g., 30):", min_value=0, max_value=120, value=30)
        duration = st.number_input("Enter duration of migraine in hours (e.g., 2.5):", min_value=0.0, value=2.5)
        frequency = st.number_input("Enter frequency of migraines per month (e.g., 3):", min_value=0, value=3)
        location = st.selectbox("Select the location of the pain:", [1, 2], format_func=lambda x: "Right" if x == 1 else "Left")
        character = st.selectbox("Select character of pain:", [1, 0], format_func=lambda x: "Throbbing" if x == 1 else "Dull")
        intensity = st.number_input("Enter intensity of pain on a scale from 1 to 10 (e.g., 8):", min_value=1, max_value=10, value=8)
        nausea = st.selectbox("Nausea?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        vomit = st.selectbox("Vomiting?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        phonophobia = st.selectbox("Phonophobia?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        photophobia = st.selectbox("Photophobia?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        visual = st.selectbox("Visual disturbances?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        sensory = st.selectbox("Sensory disturbances?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        dysphasia = st.selectbox("Dysphasia?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        dysarthria = st.selectbox("Dysarthria?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        vertigo = st.selectbox("Vertigo?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    
    with col3:
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
if selected == 'Migraine Prediction':
        features = migraine_input()
        
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


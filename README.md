# Diagnosify - Multiple Disease Prediction System

Diagnosify is a predictive healthcare tool designed to analyze a patient's health metrics and provide an indication of potential diseases. This system leverages machine learning to predict the likelihood of various diseases, including diabetes, heart disease, Parkinson's disease, COPD, and kidney disease. Diagnosify aims to be an accessible, fast, and accurate aid for both medical professionals and patients.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Setup](#setup)
- [Usage](#usage)
- [Model Information](#model-information)


## Overview
Diagnosify is built using machine learning algorithms and aims to assist healthcare professionals by providing a secondary diagnostic tool for common diseases. With a user-friendly Streamlit interface, users can enter health metrics and get a probabilistic assessment of specific diseases, aiding in early detection and preventive care.

## Features
- **Multi-Disease Prediction**: Predicts the risk of multiple diseases, such as:
  - Diabetes
  - Heart Disease
  - Parkinson's Disease
  - Chronic Obstructive Pulmonary Disease (COPD)
  - Kidney Disease
- **User-Friendly Interface**: Accessible via a web application built with Streamlit for seamless interaction.
- **Real-Time Prediction**: Provides immediate results based on input health metrics.
- **Secure and Scalable**: Designed to be deployed on Streamlit Cloud, with the ability to scale and integrate with more data.

## Technologies
- **Python**: Main programming language for building the app and models.
- **Streamlit**: Framework used to build the web application.
- **Scikit-Learn** and **Joblib**: For machine learning model building and serialization.
- **Pandas** and **NumPy**: Data processing and manipulation.
- **Jupyter Notebook**: Used during the initial model development and testing.
- **Matplotlib** and **Seaborn**: Data visualization libraries for analysis.

## Setup
### Prerequisites
- **Python 3.7 or later**: Make sure Python is installed on your system. You can download it from [here](https://www.python.org/downloads/).
- **Git**: If you want to clone the repository, install Git from [here](https://git-scm.com/).

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/Diagnosify.git
   cd Diagnosify
2. **Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate    # On Windows use: env\Scripts\activate
3. **Create a virtual environment:**
   ```bash
   pip install -r requirements.txt
4. **Run the application locally:**
   ```bash
   streamlit run disease_main.py

## Deploying on Streamlit Cloud
1. Push your code to a GitHub repository.
2. In Streamlit Cloud, create a new app and connect it to your GitHub repository.
3. Ensure your `requirements.txt` file is included in the repository for dependency installation.
4. Click "Deploy" to launch the app.

## Usage
1. **Launch the App**: Access Diagnosify locally or via Streamlit Cloud.
2. **Enter Health Metrics**: Input relevant patient data such as age, blood pressure, glucose levels, etc.
3. **Get Prediction**: The app will output a probability score for each disease.
4. **Interpret Results**: Use the prediction probabilities as supplementary information alongside professional medical advice.

## Model Information
Diagnosify uses a range of supervised machine learning models trained on curated datasets to predict each disease. Here is a quick summary of the models used:

- **Diabetes**: Random Forest Classifier
- **Heart Disease**: Logistic Regression
- **Parkinson's Disease**: Support Vector Machine
- **COPD**: Gradient Boosting Classifier
- **Kidney Disease**: Decision Tree Classifier

Each model has been tuned for accuracy and optimized for real-time prediction.



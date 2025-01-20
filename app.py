import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import shap
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/models/parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    # getting the input data from the user
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

    # code for Prediction
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

    # SHAP Explanation for the Diabetes Prediction
    if diab_diagnosis:
        explainer = shap.Explainer(diabetes_model, X_train)  # X_train should be your training data
        shap_values = explainer([user_input])

        st.subheader('SHAP Feature Importance for Diabetes Prediction')
        shap.summary_plot(shap_values, X_train)  # Show summary plot for feature importance

        st.pyplot(plt)  # Render the plot on Streamlit app

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    # Same steps for Heart Disease Prediction, with SHAP visualizations for this model as well.
    pass

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    # Same steps for Parkinson's Disease Prediction, with SHAP visualizations for this model as well.
    pass

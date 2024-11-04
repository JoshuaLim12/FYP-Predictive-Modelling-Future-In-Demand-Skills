# cd "C:\Users\Joshuaa\OneDrive\Documents\Study\FYP\Code"
# streamlit run app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the sccaler
s_scaler = joblib.load('sscaler.pkl')
mm_scaler = joblib.load('minmax_scaler_company_size.pkl')

# Load best model
model = joblib.load('logistic_regression_tuned_model.pkl')

# Define feature (Same order of X_train)
columns_order = [
    'Job Posting Date Ordinal', 'Year', 'Month', 'Day', 'Avg Salary', 'Avg Experience', 'Company Size', 
    'Job Title_Back-End Developer', 'Job Title_Business Analyst', 'Job Title_Data Analyst', 'Job Title_Data Engineer', 
    'Job Title_Data Scientist', 'Job Title_Database Administrator', 'Job Title_Database Developer', 
    'Job Title_Front-End Developer', 'Job Title_Front-End Engineer', 'Job Title_IT Administrator', 'Job Title_IT Manager', 
    'Job Title_IT Support Specialist', 'Job Title_Java Developer', 'Job Title_Network Engineer', 'Job Title_Network Security Specialist', 
    'Job Title_Project Manager', 'Job Title_Software Developer', 'Job Title_Software Engineer', 'Job Title_Software Tester', 
    'Job Title_Systems Administrator', 'Job Title_Systems Analyst', 'Job Title_Systems Engineer', 'Job Title_UI Developer', 
    'Job Title_UX Researcher', 'Job Title_Web Designer', 'Job Title_Web Developer', 'Qualifications_Bachelor', 
    'Qualifications_Doctorate', 'Qualifications_Master', 'Work Type_Contract', 'Work Type_Full-Time', 'Work Type_Intern', 
    'Work Type_Part-Time', 'Work Type_Temporary', 'Preference_Both', 'Preference_Female', 'Preference_Male'
]

# Define skills (target labels)
skills = [
    'Backend Development', 'Business Intelligence & Strategy', 'Cloud Computing & Infrastructure', 
    'Collaboration & Communication', 'Construction & Engineering', 'Data Management & Analytics', 
    'Debugging & Troubleshooting', 'DevOps & Continuous Integration', 'E-commerce & Platforms', 
    'Frontend Development', 'Graphic Design & Adobe Suite', 'Healthcare & Industry Knowledge', 
    'Machine Learning & AI', 'Mobile App Development', 'Network Administration', 'Programming', 
    'Project Management & Leadership', 'Security & Vulnerability Management', 'Soft Skills & Problem Solving', 
    'System Analysis & IT Support', 'Testing & Quality Assurance', 'UI/UX & Design', 'Web Development'
]

# Title and description & header with larger font size
st.markdown("<h1 style='font-size: 36px;'>Future In-Demand Skills Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='font-size: 24px;'>Predict which skills will be in demand based on job-related features using different models.</h3>", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("Input Job Features")

# Define the start and end dates
start_date = datetime(2021, 10, 1)
end_date = datetime(2023, 8, 31)

# Create a date slider in the sidebar
job_posting_date = st.sidebar.slider("Select any date between", start_date, end_date, step=timedelta(days=1))
                             
# Input sliders
avg_salary = st.sidebar.slider("Select Avg Salary", min_value=67500, max_value=97500, step=500)
avg_experience = st.sidebar.slider("Years of Experience", min_value=0, max_value=30, step=1)

# Company Size mapping
company_size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
company_size_input = st.sidebar.selectbox("Select Company Size", list(company_size_mapping.keys()))
company_size = company_size_mapping[company_size_input]

# Job Title selection
job_titles = st.sidebar.selectbox("Select Job Title", [
    'Job Title_Back-End Developer', 'Job Title_Business Analyst', 'Job Title_Data Analyst', 'Job Title_Data Engineer', 
    'Job Title_Data Scientist', 'Job Title_Database Administrator', 'Job Title_Database Developer', 
    'Job Title_Front-End Developer', 'Job Title_Front-End Engineer', 'Job Title_IT Administrator', 'Job Title_IT Manager', 
    'Job Title_IT Support Specialist', 'Job Title_Java Developer', 'Job Title_Network Engineer', 'Job Title_Network Security Specialist', 
    'Job Title_Project Manager', 'Job Title_Software Developer', 'Job Title_Software Engineer', 'Job Title_Software Tester', 
    'Job Title_Systems Administrator', 'Job Title_Systems Analyst', 'Job Title_Systems Engineer', 'Job Title_UI Developer', 
    'Job Title_UX Researcher', 'Job Title_Web Designer', 'Job Title_Web Developer'
])

# Qualifications selection
qualifications = st.sidebar.selectbox("Select Qualifications", ["Bachelor", "Master", "Doctorate"])

# Work Type selection
work_type = st.sidebar.selectbox("Select Work Type", ["Full-Time", "Part-Time", "Contract", "Temporary", "Intern"])

# Gender preference selection
preference = st.sidebar.selectbox("Select Your Gender", ["Male", "Female", "Both"])

# Function to preprocess the input data
def preprocess_input(job_posting_date, avg_salary, avg_experience, company_size, job_titles, qualifications, work_type, preference):

    scaled_values = s_scaler.transform(np.array([[avg_salary, avg_experience]]))

    # Preprocess the numeric and date features
    input_data = { 'Job Posting Date Ordinal': job_posting_date.toordinal(),
                  'Year': job_posting_date.year,
                  'Month': job_posting_date.month,
                  'Day': job_posting_date.day,
                  'Avg Salary': float(scaled_values[0][0]),  # Extract scaled salary
                  'Avg Experience': float(scaled_values[0][1]),  # Extract scaled experience
                  'Company Size': float(mm_scaler.transform(np.array([[company_size]]))[0][0]) }
    
    # Job Title selection then One-hot encoding
    job_titles_full_name =  [
        'Job Title_Back-End Developer', 'Job Title_Business Analyst', 'Job Title_Data Analyst', 'Job Title_Data Engineer', 
        'Job Title_Data Scientist', 'Job Title_Database Administrator', 'Job Title_Database Developer', 
        'Job Title_Front-End Developer', 'Job Title_Front-End Engineer', 'Job Title_IT Administrator', 'Job Title_IT Manager', 
        'Job Title_IT Support Specialist', 'Job Title_Java Developer', 'Job Title_Network Engineer', 'Job Title_Network Security Specialist', 
        'Job Title_Project Manager', 'Job Title_Software Developer', 'Job Title_Software Engineer', 'Job Title_Software Tester', 
        'Job Title_Systems Administrator', 'Job Title_Systems Analyst', 'Job Title_Systems Engineer', 'Job Title_UI Developer', 
        'Job Title_UX Researcher', 'Job Title_Web Designer', 'Job Title_Web Developer'
    ]
    for jt in job_titles_full_name:
        input_data[jt] = 1 if jt == job_titles else 0 

    # One-hot encoding for qualifications, work type and preference
    input_data['Qualifications_Bachelor'] = 1 if qualifications == 'Bachelor' else 0
    input_data['Qualifications_Doctorate'] = 1 if qualifications == 'Doctorate' else 0
    input_data['Qualifications_Master'] = 1 if qualifications == 'Master' else 0
    input_data['Work Type_Contract'] = 1 if work_type == 'Contract' else 0
    input_data['Work Type_Full-Time'] = 1 if work_type == 'Full-Time' else 0
    input_data['Work Type_Intern'] = 1 if work_type == 'Intern' else 0
    input_data['Work Type_Part-Time'] = 1 if work_type == 'Part-Time' else 0
    input_data['Work Type_Temporary'] = 1 if work_type == 'Temporary' else 0
    input_data['Preference_Both'] = 1 if preference == 'Both' else 0
    input_data['Preference_Female'] = 1 if preference == 'Female' else 0
    input_data['Preference_Male'] = 1 if preference == 'Male' else 0

    input_df = pd.DataFrame([input_data]) # Convert the dictionary to a DataFrame with correct column order

    try:
        input_df = input_df[columns_order]  # Ensure correct column order
    except KeyError as e:
        print(f"KeyError: {e} - Available columns: {input_df.columns.tolist()}")
    
    return input_df

# Get the preprocessed input data
input_data = preprocess_input(job_posting_date, avg_salary, avg_experience, company_size, job_titles, qualifications, work_type, preference)

# Predict button for selected model
if st.sidebar.button("Predict In-Demand Skills"):
    try:
        # Ensure input_data is a DataFrame with the correct shape
        input_data_array = input_data.values  # Convert DataFrame to a numpy array

        # Run the model prediction
        predictions = model.predict(input_data_array)

        # Check the shape of predictions
        if predictions.shape[1] != len(skills):
            st.error("Prediction output size does not match the number of skills.")
        else:
            # Map predictions to skills
            predicted_skills = [skills[i] for i in range(len(skills)) if predictions[0][i] == 1]

            # Show the predicted skills
            if predicted_skills:
                st.text_area("Predicted Skills:", value=', '.join(predicted_skills), height=100)
            else:
                st.write("No skills predicted.")

            # Or use markdown to list the skills
            st.markdown("### Skills List:")
            for skill in predicted_skills:
                st.markdown(f"- {skill}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

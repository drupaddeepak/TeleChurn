
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the pre-trained model
model = pickle.load(open(r'C:\Users\My PC\Downloads\Telechurn\model.sav', 'rb'))

# Set the title and description of the app
st.title('Customer Churn Prediction')
st.write('This app predicts whether a customer will churn or not based on their information.')

# Create the input fields for all the features
st.sidebar.header('Customer Information')

def user_input_features():
    gender = st.sidebar.radio('Gender', ('Male', 'Female'))
    senior_citizen = st.sidebar.checkbox('Senior Citizen')
    partner = st.sidebar.radio('Partner', ('Yes', 'No'))
    dependents = st.sidebar.radio('Dependents', ('Yes', 'No'))
    phone_service = st.sidebar.radio('Phone Service', ('Yes', 'No'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.radio('Paperless Billing', ('Yes', 'No'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthly_charges = st.sidebar.number_input('Monthly Charges', min_value=0.0, max_value=150.0, value=70.0)
    total_charges = st.sidebar.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=2000.0)
    tenure_group = st.sidebar.selectbox('Tenure Group (months)', ('1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72'))

    data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen else 0,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess the user input
# The model was trained on a one-hot encoded dataset.
# We need to create the same columns in the input dataframe.

# Create a copy of the input dataframe
df = input_df.copy()

# One-hot encode the categorical features
df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                 'PaperlessBilling', 'PaymentMethod', 'tenure_group'])

# The list of all columns the model was trained on
model_columns = ['Unnamed: 0', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female',
       'gender_Male', 'Partner_No', 'Partner_Yes', 'Dependents_No',
       'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes',
       'MultipleLines_No', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No',
       'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
       'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36',
       'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72']


# Align the dataframe columns
df_aligned = df.reindex(columns=model_columns, fill_value=0)
df_aligned['Unnamed: 0'] = 0


st.subheader('User Input parameters')
st.write(input_df)


# Make predictions
if st.button('Predict'):
    prediction = model.predict(df_aligned)
    prediction_proba = model.predict_proba(df_aligned)

    st.subheader('Prediction')
    churn_status = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f'Churn: {churn_status}')

    st.subheader('Prediction Probability')
    st.write(f'Probability of No Churn: {prediction_proba[0][0]:.2f}')
    st.write(f'Probability of Churn: {prediction_proba[0][1]:.2f}')

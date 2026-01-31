import streamlit as st
import joblib
import pandas as pd
import numpy as np


# We use @st.cache_resource so it only loads once (makes it faster)
@st.cache_resource
def load_model():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Title
st.title("🔮 Telco Customer Churn Predictor")
st.write("Enter customer details below to check if they are likely to cancel their service.")


col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Bill ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
    
with col2:
    fiber_optic = st.selectbox("Internet Type", ["DSL", "Fiber Optic", "No Internet"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    pay_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# Advanced Options (Hidden inside an expander to keep it clean)
with st.expander("Show Advanced Options (Demographics & Services)"):
    senior = st.checkbox("Senior Citizen")
    tech_support = st.checkbox("Has Tech Support")
    # we can add extra options too

#predict button
if st.button("Predict Churn Risk"):
    
    #Defining Columns
    columns = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 
        'gender_Male', 'Partner_Yes', 'Dependents_Yes',
        'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year',
        'PaperlessBilling_Yes',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    # initialising data with 0's
    input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    
    # User input
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges
    input_df['TotalCharges'] = total_charges
    
    if senior:
        input_df['SeniorCitizen'] = 1
        
    if fiber_optic == "Fiber Optic":
        input_df['InternetService_Fiber optic'] = 1
    elif fiber_optic == "No Internet":
        input_df['InternetService_No'] = 1
        
    if contract == "One year":
        input_df['Contract_One year'] = 1
    elif contract == "Two year":
        input_df['Contract_Two year'] = 1
        
    if pay_method == "Electronic check":
        input_df['PaymentMethod_Electronic check'] = 1
        
    if tech_support:
        input_df['TechSupport_Yes'] = 1

    # Scale and Predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    # Result
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"🚨 **HIGH RISK!** This customer is likely to churn.")
        st.write(f"Confidence: **{probability[0][1]*100:.2f}%**")
    else:
        st.success(f"✅ **SAFE!** This customer is likely to stay.")
        st.write(f"Confidence: **{probability[0][0]*100:.2f}%**")
import joblib
import pandas as pd
import numpy as np


model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

print("--- CHURN PREDICTOR ---")

#Defining columns
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

# creating fake customer details


# We start with all zeros
customer_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

# Now we fill in their specific details
customer_data['tenure'] = 72
customer_data['MonthlyCharges'] = 20.0
customer_data['TotalCharges'] = 90.5
customer_data['SeniorCitizen'] = 0
customer_data['InternetService_Fiber optic'] = 0  # Using Fiber
customer_data['Contract_Two year'] = 1 # Using Check


#scaling
customer_data_scaled = scaler.transform(customer_data)

#prediction
prediction = model.predict(customer_data_scaled)
probability = model.predict_proba(customer_data_scaled)

if prediction[0] == 1:
    print(f"\n🚨 ALERT: This customer is likely to CHURN (Leave)!")
    print(f"Risk Probability: {probability[0][1]*100:.2f}%")
else:
    print(f"\n✅ SAFE: This customer will likely STAY.")
    print(f"Safety Probability: {probability[0][0]*100:.2f}%")
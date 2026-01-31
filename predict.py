import joblib
import pandas as pd
import numpy as np

# 1. LOAD
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. DEFINE COLUMNS (Must match training EXACTLY)
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

# HELPER: Get number or 0
def get_number_or_zero(prompt):
    try:
        return float(input(prompt))
    except ValueError:
        return 0.0

# HELPER: Get Yes/No and return 1 or 0
def get_yes_no(prompt):
    ans = input(prompt + " (y/n): ").lower()
    return 1 if ans.startswith('y') else 0

# 3. ASK THE USER (Expanded List)
print("\n--- 📞 DETAILED CUSTOMER CHECK ---")
print("(Press Enter to default to 0/No)")

# -- The Big Numbers --
tenure = get_number_or_zero("Months as customer: ")
monthly_bill = get_number_or_zero("Monthly Bill ($): ")
total_bill = get_number_or_zero("Total Charges ($): ")

# -- The Crucial Categories --
# We create a blank row
input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

# Fill Numbers
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_bill
input_data['TotalCharges'] = total_bill

# Fill "Senior Citizen" (Demographics matter!)
if get_yes_no("Are they a Senior Citizen?"):
    input_data['SeniorCitizen'] = 1

# Fill "Internet Type"
net_type = input("Internet Type? (fiber/dsl/no): ").lower()
if 'fiber' in net_type:
    input_data['InternetService_Fiber optic'] = 1
elif 'no' in net_type:
    input_data['InternetService_No'] = 1

# Fill "Tech Support" (People with tech support rarely leave!)
if get_yes_no("Do they have Tech Support?"):
    input_data['TechSupport_Yes'] = 1

# Fill "Payment Method" (Electronic check users leave frequently)
pay_method = input("Payment Method? (check/card/bank): ").lower()
if 'check' in pay_method:
    # We assume 'Electronic check' as it's the most common risky one
    input_data['PaymentMethod_Electronic check'] = 1

# Fill "Contract"
contract = input("Contract Length? (month/1year/2year): ").lower()
if '2' in contract:
    input_data['Contract_Two year'] = 1
elif '1' in contract:
    input_data['Contract_One year'] = 1
# else it stays 0 (Month-to-month)

# 4. PREDICT
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
probability = model.predict_proba(input_data_scaled)

print("\n" + "="*30)
if prediction[0] == 1:
    print(f"🚨 RESULT: CHURN RISK!")
    print(f"Risk: {probability[0][1]*100:.2f}%")
else:
    print(f"✅ RESULT: SAFE.")
    print(f"Safety: {probability[0][0]*100:.2f}%")
print("="*30 + "\n")
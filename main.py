import pandas as pd
import numpy as np


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.dtypes)
#changing TotalCharges column to int form object
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
#check for Nan
df['TotalCharges'].isna().sum()
df['TotalCharges'] = df['TotalCharges'].fillna(0)


df.drop(columns=['customerID'], inplace=True)

# Yes vs No
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(5,5))
sns.countplot(x='Churn', data=df)
plt.title('Churn vs Non-Churn')
plt.show()
# Yes -> 1(churn)
# No  -> 0 
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Automatic encoding for everything else
# 'get_dummies' turns text columns into binary columns (0s and 1s).
# 'drop_first=True' is a trick to avoid redundancy (ex: if you have Male/Female, you only need 'Male' column. If Male=0, we know they are Female).
df = pd.get_dummies(df, drop_first=True,dtype = int)

# new data
print("New Data Shape:", df.shape)
print(df.head())

# correlation with churn
print("\n--- Top 5 Factors Driving Churn ---")
print(df.corr()['Churn'].sort_values(ascending=False).head(5))

# model training
X = df.drop('Churn', axis=1)
y = df['Churn']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# scaling for Logistic Regression model
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# m1 Logistic Regression
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

# m2 Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train) # Random Forest doesn't need scaling!
rf_pred = rf_model.predict(X_test)

#result
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_pred):.4f}")
print(f"Random Forest Accuracy:     {accuracy_score(y_test, rf_pred):.4f}")


print("\n--- Random Forest Detail Report ---")
print(classification_report(y_test, rf_pred))

#logistic was better

#Confusion matrix for logistic regression
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, log_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()
#Feature importance
weights = pd.Series(log_model.coef_[0], index=X.columns) #weights of log_model

print("\n--- Top Factors Increasing Churn ---")
print(weights.sort_values(ascending=False).head(5))

print("\n--- Top Factors PREVENTING Churn ---")
print(weights.sort_values(ascending=True).head(5))

# Plot of tp 10 churn factors
plt.figure(figsize=(8,6))
weights.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title('Top 10 Features Driving Churn')
plt.show()

#final comaprison using xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
#still logistic model gives best result

#saving our final logistic regression model
import joblib

# We also save the 'scaler' so we can translate future data correctly.

model_filename = 'churn_model.pkl'
scaler_filename = 'scaler.pkl'

joblib.dump(log_model, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"\n--- SUCCESS ---")
print(f"Model saved as: {model_filename}")
print(f"Scaler saved as: {scaler_filename}")
print("You can now use these files to predict churn for new customers!")
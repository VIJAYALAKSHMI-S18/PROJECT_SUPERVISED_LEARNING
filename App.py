import streamlit as st
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------- TITLE ----------------
st.title("Customer Satisfaction Predictor")

# ---------------- LOAD & PREPARE DATA (BACKEND ONLY) ----------------
df = pd.read_csv("Customer_Satisfaction.csv")

df.drop(["Unnamed: 0", "id"], axis=1, inplace=True, errors="ignore")
df.fillna(df.median(numeric_only=True), inplace=True)

label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# ---------------- USER INPUT SECTION ----------------
st.subheader("Enter Customer Details")

user_input = {}

for col in X.columns:
    if col in label_encoders:
        user_input[col] = st.selectbox(col, label_encoders[col].classes_)
    else:
        user_input[col] = st.slider(col, float(X[col].min()), float(X[col].max()))

# ---------------- PREDICT BUTTON ----------------
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Customer is SATISFIED")
    else:
        st.error("❌ Customer is NOT SATISFIED")

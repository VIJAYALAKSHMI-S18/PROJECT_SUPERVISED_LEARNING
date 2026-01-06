import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Customer Satisfaction Predictor")

# ---------------- LOAD & TRAIN (BACKEND) ----------------
@st.cache_resource
def load_model():
    df = pd.read_csv("Customer_Satisfaction.csv")

    df.drop(["Unnamed: 0", "id"], axis=1, inplace=True, errors="ignore")
    df.fillna(df.median(numeric_only=True), inplace=True)

    label_encoders = {}

    # Encode ONLY feature columns
    for col in df.select_dtypes(include="object").columns:
        if col != "satisfaction":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    X = df.drop("satisfaction", axis=1)
    y = df["satisfaction"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler, label_encoders, X

model, scaler, label_encoders, X = load_model()

# ---------------- UI ----------------
st.title("Customer Satisfaction Predictor")
st.write("Fill in customer details and click Predict")

user_input = {}

for col in X.columns:
    if col in label_encoders:
        user_input[col] = st.selectbox(col, label_encoders[col].classes_, key=col)
    else:
        user_input[col] = st.number_input(
            col,
            value=float(X[col].mean()),
            key=col
        )

# ---------------- PREDICT ----------------
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical inputs safely
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Customer is SATISFIED")
    else:
        st.error("❌ Customer is NOT SATISFIED")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# ---------------- TITLE ----------------
st.title("Customer Satisfaction ML Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Customer_Satisfaction.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- DATA CLEANING ----------------
df.drop(["Unnamed: 0", "id"], axis=1, inplace=True, errors="ignore")
df.fillna(df.median(numeric_only=True), inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ---------------- INPUT / OUTPUT ----------------
X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Naive Bayes": GaussianNB(),
    "Bagging": BaggingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

accuracy = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy[name] = accuracy_score(y_test, preds)
    predictions[name] = preds
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")

# ---------------- ACCURACY BAR CHART (OUTPUT) ----------------
st.subheader("Model Accuracy Comparison (Output)")

fig, ax = plt.subplots()
ax.bar(accuracy.keys(), accuracy.values())
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------------- ACTUAL vs PREDICTED BAR ----------------
st.subheader("Actual vs Predicted Output (Logistic Regression)")

actual_counts = np.bincount(y_test)
pred_counts = np.bincount(predictions["Logistic Regression"])

labels = ["Class 0", "Class 1"]

x = np.arange(len(labels))
width = 0.35

fig2, ax2 = plt.subplots()
ax2.bar(x - width/2, actual_counts, width, label="Actual")
ax2.bar(x + width/2, pred_counts, width, label="Predicted")

ax2.set_xlabel("Classes")
ax2.set_ylabel("Count")
ax2.set_title("Actual vs Predicted Satisfaction")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()

st.pyplot(fig2)

# ---------------- PCA INPUT VISUAL ----------------
st.subheader("Input Feature Distribution (PCA)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig3, ax3 = plt.subplots()
ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.6)
ax3.set_xlabel("Principal Component 1")
ax3.set_ylabel("Principal Component 2")
ax3.set_title("Input Features after PCA")
st.pyplot(fig3)

st.success("Models trained, outputs visualized, and models saved successfully!")

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"

st.set_page_config(
    page_title="Spam Classification Dashboard",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Email Spam Classification Dashboard")
st.write(
    "This application compares machine-learning classifiers trained on the "
    "UCI Spambase dataset. Upload the generated test_data.csv file and choose "
    "a model to view its evaluation metrics."
)

model_paths = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "kNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}
feature_path = MODEL_DIR / "feature_columns.joblib"

missing = [str(p.name) for p in list(model_paths.values()) + [feature_path] if not p.exists()]
if missing:
    st.error(
        "The trained model files are not present yet. Run model/ML_Assignment_2.ipynb "
        "or model/train_models.py first."
    )
    st.write("Missing files:", missing)
    st.stop()

@st.cache_resource
def load_feature_columns():
    return joblib.load(feature_path)

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

feature_columns = load_feature_columns()

st.sidebar.header("Model Evaluation Controls")
selected_model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(model_paths.keys())
)
st.sidebar.info("0 = Not Spam\n\n1 = Spam")

st.subheader("1. Upload Test Dataset")
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload the generated test_data.csv file to begin evaluation.")
    st.stop()

data = pd.read_csv(uploaded_file)
st.success("Test dataset uploaded successfully.")
st.write("Dataset shape:", data.shape)
st.subheader("Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

if "target" not in data.columns:
    st.error("The uploaded CSV must contain a 'target' column.")
    st.stop()

missing_features = [f for f in feature_columns if f not in data.columns]
if missing_features:
    st.error("Uploaded data is missing required features.")
    st.write(missing_features)
    st.stop()

X_test = data[feature_columns]
y_test = data["target"]

model = load_model(str(model_paths[selected_model]))
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

try:
    auc = roc_auc_score(y_test, y_prob)
except ValueError:
    auc = np.nan

st.subheader("2. Selected Model")
st.info(selected_model)

st.subheader("3. Evaluation Metrics")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Accuracy", f"{accuracy:.4f}")
    st.metric("Precision", f"{precision:.4f}")
with c2:
    st.metric("AUC", "N/A" if np.isnan(auc) else f"{auc:.4f}")
    st.metric("Recall", f"{recall:.4f}")
with c3:
    st.metric("F1 Score", f"{f1:.4f}")
    st.metric("MCC Score", f"{mcc:.4f}")

st.subheader("4. Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
image = ax.imshow(cm)
ax.set_title(f"Confusion Matrix - {selected_model}")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Not Spam", "Spam"])
ax.set_yticklabels(["Not Spam", "Spam"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.colorbar(image, ax=ax)
st.pyplot(fig)

st.subheader("5. Classification Report")
report = classification_report(
    y_test, y_pred,
    target_names=["Not Spam", "Spam"],
    output_dict=True,
    zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader("6. Prediction Results")
prediction_results = data.copy()
prediction_results["Predicted_Class"] = y_pred
prediction_results["Spam_Probability"] = y_prob
prediction_results["Prediction_Label"] = prediction_results["Predicted_Class"].map(
    {0: "Not Spam", 1: "Spam"}
)
st.dataframe(prediction_results.head(20), use_container_width=True)

csv_output = prediction_results.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Prediction Results",
    data=csv_output,
    file_name="prediction_results.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Machine Learning Assignment 2 | UCI Spambase Classification Project")
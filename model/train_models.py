from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)

# ------------------------------------------------------------
# Locate project folder robustly
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parent.parent
MODEL_DIR = PROJECT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Project folder:", PROJECT_DIR)

# ------------------------------------------------------------
# Dataset columns
# ------------------------------------------------------------
feature_names = [
    "word_freq_make","word_freq_address","word_freq_all","word_freq_3d",
    "word_freq_our","word_freq_over","word_freq_remove","word_freq_internet",
    "word_freq_order","word_freq_mail","word_freq_receive","word_freq_will",
    "word_freq_people","word_freq_report","word_freq_addresses","word_freq_free",
    "word_freq_business","word_freq_email","word_freq_you","word_freq_credit",
    "word_freq_your","word_freq_font","word_freq_000","word_freq_money",
    "word_freq_hp","word_freq_hpl","word_freq_george","word_freq_650",
    "word_freq_lab","word_freq_labs","word_freq_telnet","word_freq_857",
    "word_freq_data","word_freq_415","word_freq_85","word_freq_technology",
    "word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct",
    "word_freq_cs","word_freq_meeting","word_freq_original","word_freq_project",
    "word_freq_re","word_freq_edu","word_freq_table","word_freq_conference",
    "char_freq_semicolon","char_freq_parenthesis","char_freq_square_bracket",
    "char_freq_exclamation","char_freq_dollar","char_freq_hash",
    "capital_run_length_average","capital_run_length_longest",
    "capital_run_length_total"
]
columns = feature_names + ["target"]

# ------------------------------------------------------------
# Load UCI Spambase dataset
# ------------------------------------------------------------
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

local_candidates = [
    PROJECT_DIR / "spambase.data",
    MODEL_DIR / "spambase.data",
]

df = None
try:
    print("Trying to download UCI Spambase dataset...")
    df = pd.read_csv(dataset_url, header=None, names=columns)
    print("Dataset downloaded successfully.")
except Exception as e:
    print("Direct download failed:", e)
    for candidate in local_candidates:
        if candidate.exists():
            df = pd.read_csv(candidate, header=None, names=columns)
            print("Loaded local dataset:", candidate)
            break

if df is None:
    raise FileNotFoundError(
        "\nCould not download Spambase.\n"
        "Please download 'spambase.data' from UCI and place it in the main "
        "ML_Assignment_2 folder, then run this file again.\n"
        f"Expected path: {PROJECT_DIR / 'spambase.data'}"
    )

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------
print("\nDataset shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("\nTarget distribution:")
print(df["target"].value_counts())

assert df.shape[0] >= 500, "Dataset must have at least 500 instances."
assert (df.shape[1] - 1) >= 12, "Dataset must have at least 12 features."

# ------------------------------------------------------------
# Train/test split
# ------------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)

# Save test data for Streamlit
test_data = X_test.copy()
test_data["target"] = y_test.to_numpy()
test_data.to_csv(PROJECT_DIR / "test_data.csv", index=False)
joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
}

model_files = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "kNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

# ------------------------------------------------------------
# Train and evaluate
# ------------------------------------------------------------
results = []
trained_models = {}

for model_name, model in models.items():
    print("\n" + "=" * 70)
    print("Training:", model_name)
    print("=" * 70)

    model.fit(X_train, y_train)
    trained_models[model_name] = model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "ML Model Name": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)

    print("Accuracy :", round(metrics["Accuracy"], 4))
    print("AUC      :", round(metrics["AUC"], 4))
    print("Precision:", round(metrics["Precision"], 4))
    print("Recall   :", round(metrics["Recall"], 4))
    print("F1       :", round(metrics["F1"], 4))
    print("MCC      :", round(metrics["MCC"], 4))

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, target_names=["Not Spam", "Spam"], zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, model_files[model_name])
    print("Saved:", model_files[model_name].name)

# ------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------
results_df = pd.DataFrame(results)
metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
results_df[metric_cols] = results_df[metric_cols].round(4)

results_path = PROJECT_DIR / "model_comparison_results.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 90)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 90)
print(results_df.to_string(index=False))

# Pick winner using F1 score for balanced spam detection performance
best_row = results_df.loc[results_df["F1"].idxmax()]
best_model_name = best_row["ML Model Name"]

print("\nOverall winner based on F1 Score:", best_model_name)
print("Winner F1:", best_row["F1"])

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("FINAL ASSIGNMENT VERIFICATION")
print("=" * 90)
print("Instances >= 500:", df.shape[0] >= 500)
print("Features >= 12  :", X.shape[1] >= 12)
print("Models trained  :", len(trained_models))
print("Metrics used    :", len(metric_cols))
print("test_data.csv   :", (PROJECT_DIR / "test_data.csv").exists())
print("results.csv     :", results_path.exists())

for name, path in model_files.items():
    print(f"{name:22s}:", path.exists())

print("\nTraining completed successfully.")
print("Next: run Streamlit from the MAIN project folder with:")
print("streamlit run app.py")
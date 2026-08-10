# Machine Learning Assignment 2

## Email Spam Classification Using Machine Learning

### A. Problem Statement

Email spam detection is a binary classification problem in which an email is classified as either spam or non-spam.

The objective of this project is to implement and compare multiple machine learning classification algorithms on the UCI Spambase dataset and evaluate their performance using different classification metrics.

The models are evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

### B. Dataset Description

**Dataset:** UCI Spambase Dataset  
**Source:** UCI Machine Learning Repository  
**Problem Type:** Binary Classification  
**Number of Instances:** 4,601  
**Number of Input Features:** 57  
**Target Classes:**

- `0` = Not Spam
- `1` = Spam

The dataset contains different attributes extracted from emails, including word frequencies, character frequencies and capital-letter sequence information.

The dataset satisfies the assignment requirements of having at least 500 instances and at least 12 features.

---

### C. GitHub Repository Link

[GitHub Repository - ML Assignment 2](https://github.com/lakshmi-manasa-n/ML_Assignment_2)

---

### D. Models Used

The following machine learning classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

---

## Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9110 | 0.9078 | 0.8828 | 0.8926 | 0.8877 | 0.8140 |
| kNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8339 | 0.9449 | 0.7178 | 0.9532 | 0.8189 | 0.6941 |
| Random Forest | 0.9457 | 0.9833 | 0.9510 | 0.9091 | 0.9296 | 0.8860 |

---

## Model Performance Observations

### Logistic Regression

Logistic Regression achieved an accuracy of **0.9294** and an AUC score of **0.9702**. It demonstrated balanced classification performance with a precision of **0.9209**, recall of **0.8981**, F1 score of **0.9093**, and MCC score of **0.8518**. Overall, Logistic Regression performed strongly on the Spambase dataset.

### Decision Tree

Decision Tree achieved an accuracy of **0.9110** and an AUC score of **0.9078**. Its precision, recall and F1 score were **0.8828**, **0.8926**, and **0.8877**, respectively. The model performed reasonably well, although its overall performance was lower than Logistic Regression and Random Forest.

### k-Nearest Neighbors (kNN)

kNN achieved an accuracy of **0.9077** and an AUC score of **0.9506**. It obtained a precision of **0.8861**, recall of **0.8788**, F1 score of **0.8824**, and MCC score of **0.8065**. The model provided good discrimination between spam and non-spam emails but performed slightly below Logistic Regression and Random Forest.

### Naive Bayes

Naive Bayes achieved an accuracy of **0.8339** and an AUC score of **0.9449**. It produced the highest recall among all the models at **0.9532**, meaning it was able to identify a very large proportion of spam emails.

However, its precision was comparatively lower at **0.7178**, indicating that it generated more false-positive spam predictions. Its F1 score was **0.8189** and MCC was **0.6941**.

### Random Forest

Random Forest achieved the best overall performance with an accuracy of **0.9457**, AUC score of **0.9833**, precision of **0.9510**, recall of **0.9091**, F1 score of **0.9296**, and MCC score of **0.8860**.

It achieved the highest Accuracy, AUC, Precision, F1 Score and MCC among the implemented models, demonstrating strong and balanced classification performance.

---

## Overall Winner

**Random Forest** is selected as the overall best-performing model for the UCI Spambase dataset.

It achieved the highest values for:

- Accuracy: **0.9457**
- AUC: **0.9833**
- Precision: **0.9510**
- F1 Score: **0.9296**
- MCC: **0.8860**

Although Naive Bayes achieved the highest Recall of **0.9532**, Random Forest provided the strongest overall balance across the evaluation metrics and is therefore selected as the best model for this dataset.

---

## Streamlit Application

The deployed Streamlit application can be accessed using the following link:

[Live Streamlit Application](https://2025ac05544-ml-assignment.streamlit.app/)

The Streamlit application provides the following functionality:

- Upload test data in CSV format
- Select a machine learning model using a dropdown
- Display Accuracy
- Display AUC Score
- Display Precision
- Display Recall
- Display F1 Score
- Display MCC Score
- Display Confusion Matrix
- Display Classification Report
- Display model prediction results

---

## Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
├── model_comparison_results.csv
│
└── model/
    ├── ML_Assignment_2.ipynb
    ├── train_models.py
    ├── feature_columns.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

---

## How to Run the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate the machine learning models:

```bash
python model/train_models.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Conclusion

Five machine learning classification algorithms were implemented and evaluated on the UCI Spambase dataset.

Among the evaluated models, **Random Forest produced the strongest overall classification performance**, while **Naive Bayes achieved the highest recall**.

The trained models were integrated into an interactive Streamlit application that allows users to upload test data, select a classification model and view the corresponding evaluation results.

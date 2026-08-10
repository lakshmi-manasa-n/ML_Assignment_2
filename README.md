# Machine Learning Assignment 2
## Email Spam Classification Using Machine Learning

### A. Problem Statement
The objective of this project is to classify email messages as spam or non-spam
and compare multiple machine-learning classification algorithms.

### B. Dataset Description
- Dataset: UCI Spambase
- Problem type: Binary classification
- Instances: 4,601
- Input features: 57
- Target: 0 = Not Spam, 1 = Spam

### C. GitHub Repository Link
Add your GitHub repository link here after uploading the project.

### D. Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest Classifier

### Required Evaluation Metrics
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison
After running the training notebook, copy the values from
`model_comparison_results.csv` into the table below.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest | | | | | | |

### Model Performance Observations
Fill these using your actual outputs after running the notebook.

- Logistic Regression:
- Decision Tree:
- kNN:
- Naive Bayes:
- Random Forest:
- Overall winner:

### Streamlit Application
Add your Streamlit Community Cloud URL here after deployment.

The application includes:
- CSV test-data upload
- Model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Prediction results

### How to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python model/train_models.py
```

Run Streamlit:

```bash
streamlit run app.py
```
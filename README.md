# ❤️ Heart Disease Detection using Machine Learning

This project aims to predict whether a person is likely to have heart disease using various machine learning classification models. 
The dataset contains 13 medical attributes such as age, sex, chest pain type, cholesterol level, etc.

## 🔍 Overview

This notebook explores and compares multiple classification models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVC)
- Naive Bayes
- Random Forest Classifier
- Gradient Boosting Classifier

We also performed:
- Feature scaling
- Hyperparameter tuning using GridSearchCV
- Feature importance visualization
- Final model evaluation using accuracy, confusion matrix, and classification report

---

## 📁 Project Structure

```bash
heart_disease_detection/
├── heart_disease_detection.ipynb      # Main project notebook
├── notes.pdf                          # Handwritten notes (optional)
└── README.md                          # Project documentation

## 📊 Dataset

The dataset includes the following features:

- `age`: Age of the person
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type (0–3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results (0–2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: The slope of the peak exercise ST segment
- `ca`: Number of major vessels (0–3) colored by fluoroscopy
- `thal`: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)
- `target`: Diagnosis of heart disease (1 = has disease, 0 = no disease)

> **Dataset Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---

## ✅ Final Model

After trying multiple models, the **Random Forest Classifier** gave the best results in terms of performance and interpretability.

### 🔧 Tuned Hyperparameters with `GridSearchCV`:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

Feature importance analysis was then used to determine which features had the biggest impact on prediction.

---

## 🔎 Insights

- Top 4 most important features:
  - `cp` (chest pain type)
  - `thalach` (max heart rate)
  - `ca` (major vessels)
  - `oldpeak` (ST depression)
  
These features alone gave fairly good results when used for training, making the model simpler and faster.

---



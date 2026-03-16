# 📉 Customer Churn Prediction

A production-ready machine learning web application that predicts whether a telecom customer will churn, built with XGBoost, FastAPI, and Docker — deployed live on Hugging Face Spaces.

🔗 **Live Demo:** [Click Here](https://huggingface.co/spaces/Mehreen1103/customer-churn-prediction)

---

## 📌 Project Overview

Customer churn is one of the biggest challenges in the telecom industry. This project builds an end-to-end ML pipeline that:
- Analyzes customer behavior and service usage patterns
- Predicts the likelihood of a customer leaving
- Serves predictions via a REST API with an interactive web UI

---
📝**Kaggle Notebook Link:** [Click Here](https://www.kaggle.com/code/mehreenrahman/telco-customer-churn)
## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| Recall (Churn class) | ~80% |
| ROC-AUC | ~85% |
| Threshold | 0.3 (optimized for recall) |

---

## 🗂️ Dataset

- **Source:** Telco Customer Churn — IBM Sample Dataset (Kaggle)
- **Size:** 7,043 customers, 21 features
- **Target:** Churn (Yes / No)
- **Class imbalance:** ~26% churn rate — handled with class_weight='balanced' and custom threshold tuning

---

## 🧠 ML Pipeline

### 1. Exploratory Data Analysis
- Correlation heatmap of all features vs Churn
- Identified top drivers: Contract type, tenure, Monthly Charges, Internet Service

### 2. Feature Engineering
- Binary encoding for Yes/No columns
- One-hot encoding for multi-class categorical columns
- Collapsed redundant No internet service flags into a single column

### 3. Models Compared
| Model | Notes |
|-------|-------|
| Random Forest | Baseline |
| LightGBM | Strong performance |
| XGBoost ✅ | Best — selected for deployment |

### 4. Hyperparameter Tuning
- Used Optuna (30 trials) to maximize recall
- Tracked experiments with MLflow

### 5. Threshold Optimization
- Optimized to 0.3 using Precision-Recall curve analysis

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost + Scikit-learn |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Frontend | HTML + CSS + JavaScript |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |

---

## 🚀 Run Locally

### With Python
```bash
git clone https://github.com/Mehreen1103/-Customer-Churn-Prediction
cd Customer-Churn-Prediction
pip install -r requirements.txt
uvicorn app:app --reload
```

### With Docker
```bash
docker build -t churn-app .
docker run -p 8000:8000 churn-app
```

---

## 📁 Project Structure
```
customer-churn-prediction/
├── app.py               # FastAPI backend with preprocessing logic
├── index.html           # Frontend UI
├── churn_model.pkl      # Trained XGBoost model
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
└── README.md
```

---

## 📊 Key Findings

- Customers on month-to-month contracts churn at 3x the rate of those on 2-year contracts
- Fiber optic internet users churn more than DSL users despite higher speeds
- Customers with tenure less than 12 months are the highest risk group
- Electronic check payment method correlates strongly with churn

---

## 👩‍💻 Author

**Mehreen**
- GitHub: [@Mehreen1103](https://github.com/Mehreen1103)
- Hugging Face: [@Mehreen1103](https://huggingface.co/Mehreen1103)

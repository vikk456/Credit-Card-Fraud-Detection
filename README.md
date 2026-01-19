# 💳 Credit Card Fraud Detection using KMeans and AutoEncoder

This project explores two unsupervised machine learning methods to detect fraud in credit card transactions:

1. **KMeans Clustering**
2. **Deep AutoEncoder**

Both models are applied to the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and evaluated based on their ability to identify fraudulent behavior without relying on labeled data during training.

---

## 📂 Dataset

- **Source:** Kaggle – Credit Card Fraud Detection
- **Samples:** 284,807 transactions
- **Features:** 30 anonymized features (`V1`–`V28`, `Time`, `Amount`)
- **Label:** `Class` (0 = normal, 1 = fraud)

---

## 🧠 Models Used

### 1. 🌀 KMeans Clustering
- Unsupervised clustering into 2 groups (fraud vs. normal)
- Scaled features using `StandardScaler`
- Evaluation based on label alignment post clustering
- Limitation: Poor performance due to overlapping clusters

### 2. 🧠 AutoEncoder Neural Network
- Trained only on **normal transactions**
- Encoder compresses input → bottleneck → decoder reconstructs it
- Fraud detected using **reconstruction error threshold**
- Achieved **60% fraud recall**

---

## 🚀 Results (Best - AutoEncoder)

| Metric        | Value |
|---------------|-------|
| **Recall (fraud)** | 0.60 |
| **Precision (fraud)** | 0.21 |
| **F1-score (fraud)** | 0.31 |
| **Accuracy (overall)** | ≈ 1.00 (due to class imbalance) |# 💳 Credit Card Fraud Detection (Modern Machine Learning Approach)

## 📌 Project Overview
Credit card fraud detection is a **highly imbalanced classification problem**, where fraudulent transactions represent a very small fraction of total transactions.  
This project implements a **modern supervised machine learning pipeline** to accurately detect fraudulent transactions using imbalance handling techniques and appropriate evaluation metrics.

This version improves upon older clustering-based approaches by using **industry-standard models** and workflows.

---

## 🎯 Objectives
- Detect fraudulent transactions with high recall
- Handle severe class imbalance effectively
- Use proper evaluation metrics instead of accuracy alone
- Build a clean and extensible ML pipeline

---

## 📂 Dataset
- **Source**: Kaggle – Credit Card Fraud Detection Dataset  
- **Total transactions**: 284,807  
- **Fraud cases**: 492 (≈ 0.17%)  
- **Features**:
  - `V1` – `V28` (PCA-transformed features)
  - `Time`
  - `Amount`
  - `Class` (0 = Legitimate, 1 = Fraud)

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Separated features and target (`Class`)
- Applied **StandardScaler** for feature normalization
- Performed **stratified train-test split**

---

### 2️⃣ Handling Class Imbalance
To address the extreme imbalance:
- Used **SMOTE (Synthetic Minority Oversampling Technique)**
- Applied **class weights** during model training
- Tuned decision thresholds to improve fraud recall

---

### 3️⃣ Models Used
- **Random Forest Classifier**
- *(Optional)* Gradient Boosting / XGBoost

Supervised learning models outperform unsupervised clustering methods for fraud detection.

---

### 4️⃣ Model Evaluation
Accuracy is misleading for fraud detection. Therefore, the following metrics are used:
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

Primary focus is placed on **Recall of the Fraud Class**.

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- *(Optional)* XGBoost, FastAPI, Streamlit

---

## 📁 Project Structure
Credit-Card-Fraud-Detection/
│
├── data/
│ └── creditcard.csv
│
├── notebooks/
│ └── Credit_Card_Detection(latest).ipynb
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ └── evaluate.py
│
├── models/
│ └── model.pkl
│
├── requirements.txt
└── README.md


---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
jupyter notebook notebooks/CreditCardAnamoly.ipynb
---

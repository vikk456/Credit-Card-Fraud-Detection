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
| **Accuracy (overall)** | ≈ 1.00 (due to class imbalance) |

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/CreditCardAnomaly.git
cd CreditCardAnomaly
pip install -r requirements.txt

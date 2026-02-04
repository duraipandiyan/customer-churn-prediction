# 📉 Customer Churn Prediction – End‑to‑End Machine Learning Project

Deployed web app: https://churn-prediction-app-qgby.onrender.com/predictdata

Predict customer churn using behavior-based machine learning and threshold optimization

## 🔍 Project Overview

Customer churn directly impacts revenue and long‑term business sustainability. This project focuses on **predicting customer churn using behavioral and engagement data**, moving beyond surface‑level satisfaction metrics to identify *early warning signals* of disengagement.

The project covers the **complete ML lifecycle**:

* Raw transactional data analysis
* Feature engineering & aggregation
* Exploratory Data Analysis (EDA)
* Model building & evaluation
* Threshold optimization
* Model deployment using Flask

This repository demonstrates how churn can be effectively predicted using **customer behavior patterns**, making it suitable for **real‑world business applications**.

---

## 🧾 Business Problem

### 📊 Dataset Overview & Class Distribution

**Raw Transaction-Level Dataset**

* Total records: **17,049 transactions**
* Time period: **January 2023 – March 2024**
* Features: **18 columns** (demographics, transactions, engagement)

**Target Distribution (Transaction Level):**

* Non‑churn (0): **15,038**
* Churn (1): **2,010**

> ⚠️ The raw dataset is **highly imbalanced**, with churned customers forming a small minority.

---

**After Aggregation (Customer Level Dataset)**

* Total records: **~5,000 customers**
* Features: **10 engineered features**

**Target Distribution (Customer Level):**

* Churn (1): **~50.96%**
* Non‑churn (0): **~49.04%**

> ✅ Aggregation naturally balances the dataset, making it more suitable for churn modeling without aggressive resampling techniques.

---

The objective of this project is:

> **Predict whether a customer will churn (not return) based on historical behavior, satisfaction indicators, and transaction patterns.**

Businesses often rely on customer ratings or feedback to understand churn. However, customers may provide positive ratings yet silently disengage over time.

**Goal:**

> Predict whether a customer is likely to churn based on historical behavior, enabling proactive retention strategies.

---

## 📊 Dataset Description

### Original Dataset (Transaction‑Level)

Each row represents a single transaction:

| Feature                  | Description                |
| ------------------------ | -------------------------- |
| Order_ID                 | Unique order identifier    |
| Customer_ID              | Unique customer identifier |
| Date                     | Transaction date           |
| Age                      | Customer age               |
| Gender                   | Customer gender            |
| City                     | Customer city              |
| Product_Category         | Purchased product category |
| Unit_Price               | Price per unit             |
| Quantity                 | Units purchased            |
| Discount_Amount          | Discount applied           |
| Total_Amount             | Final order value          |
| Payment_Method           | Payment type               |
| Device_Type              | Device used                |
| Session_Duration_Minutes | Session length             |
| Pages_Viewed             | Pages viewed               |
| Delivery_Time_Days       | Delivery time              |
| Customer_Rating          | Rating provided            |

---

## 🔄 Feature Engineering & Aggregation

To enable customer‑level churn prediction, transactional data was aggregated into **customer‑level behavioral features**.

### Aggregated Features

| Feature              | Description                                 |
| -------------------- | ------------------------------------------- |
| Total_Orders         | Total number of orders                      |
| Total_Spend          | Cumulative spending                         |
| Total_Quantity       | Total quantity purchased                    |
| Total_Discount       | Total discounts received                    |
| Avg_Session_Duration | Average session duration                    |
| Avg_Pages_Viewed     | Average pages viewed                        |
| Avg_Delivery_Time    | Average delivery time                       |
| Avg_Rating           | Average customer rating                     |
| Recency_Days         | Days since last purchase                    |
| Churn                | Target variable (1 = churned, 0 = retained) |

This transformation converts **event‑level data into behavioral intelligence**.

---

## 📈 Exploratory Data Analysis (EDA)

### 🔹 Churn vs Recency

**Key Insight:**

* Churned customers show **significantly higher recency**
* Avg. recency:

  * Churned: ~200 days
  * Retained: ~40 days

➡️ **Interpretation:**
Customers who have not interacted recently are far more likely to churn.

---

### 🔹 Churn vs Total Orders & Total Spend

**Insight – Purchase Behavior:**

* Churned customers place fewer orders
* They contribute significantly lower total spend

**Actionable Business Angle:**

* Identify low‑frequency, low‑spend customers early
* Trigger targeted campaigns:

  * Discounts
  * Loyalty offers
  * Re‑engagement reminders

---

### 🔹 Churn vs Average Rating

**Insight – Customer Satisfaction:**

* Average ratings are **nearly identical** for churned and retained customers

**Interpretation:**

* Positive ratings do **not guarantee retention**
* Churn is more strongly driven by *behavior*, not explicit feedback

**Business Implication:**

> Relying solely on ratings to predict churn can be misleading.

---

## 🧠 Key EDA Findings

* ✅ **Recency is the strongest churn indicator**
* ✅ Lower engagement (orders & spend) correlates with churn
* ❌ Customer ratings alone are weak predictors

---

## 🧪 Modeling Approach

### Preprocessing

* Missing value imputation (most frequent)
* Feature scaling using StandardScaler
* Target leakage prevented by excluding Customer_ID, Recency, and Churn from inputs

### Models Evaluated

* Logistic Regression
* Random Forest
* Gradient Boosting
* AdaBoost
* K‑Nearest Neighbors
* Decision Tree

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1‑Score
* ROC‑AUC

Threshold tuning (0.3–0.6) was applied to optimize business trade‑offs.

---

## 🏆 Final Model Selection

**Chosen Model:** Logistic Regression (Tuned)

**Why Logistic Regression?**

* Best balance of Precision, Recall, and F1‑Score
* Strong ROC‑AUC
* Highly interpretable (important for business decisions)
* Stable predictions across thresholds

**Final Decision Threshold:** 0.4

---

## 🚀 Deployment

The model is deployed using **Flask**, enabling real‑time churn prediction through a web interface.

### Application Flow

1. User inputs customer behavior data
2. Data is preprocessed using saved pipeline
3. Model predicts churn probability
4. Threshold applied → churn / no churn
5. Result displayed in UI

---

## 🖥️ Tech Stack

* Python
* Pandas, NumPy
* Scikit‑learn
* Flask
* HTML & CSS
* Dill (model serialization)

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── exception.py
│   └── logger.py
│
├── templates/
│   └── home.html
│
├── app.py
└── README.md
```

---

## 💡 Business Value

* Enables **proactive churn prevention**
* Focuses on **behavioral signals**, not assumptions
* Can be integrated into CRM systems for real‑time alerts

---

## 📌 Future Enhancements

* Cost‑sensitive learning
* Customer lifetime value integration
* Model explainability using SHAP
* Cloud deployment (Render / AWS)

---

## 👤 Author

**Durai Pandiyan**

This project demonstrates practical ML engineering skills, business understanding, and deployment readiness.

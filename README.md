# DES-Thermodynamic-properties-prediction-using-DL-and-ML-comparative-analysis
# Thermodynamic Properties Prediction: Molar Volume and Cohesive Energy

## 📘 Project Overview

This project focuses on predicting **molar volume** and **cohesive energy** of chemical compounds using machine learning and deep learning models. Accurate prediction of these thermodynamic properties is vital for material design, molecular simulations, and understanding intermolecular interactions.

## 🧪 Dataset Description

* **Features:**

  * Molecular descriptors (calculated using Mordred or RDKit)

* **Targets:**

  * **Molar Volume (cm³/mol)**
  * **Cohesive Energy (kJ/mol)**

* **Preprocessing Steps:**

  * SMILES standardization
  * Descriptor generation
  * Removal of non-numeric and missing-value descriptors
  * Feature scaling (MinMax or StandardScaler)

## 🧬 Molecular Descriptor Calculation

* **Tools Used:** Mordred / RDKit
* **Steps:**

  * Convert SMILES to molecular descriptors
  * Filter out non-numeric and missing-value columns
  * Save clean feature matrix for modeling

## 📊 Feature Selection Techniques

To enhance model performance and reduce overfitting:

* Variance Threshold
* PCA (Principal Component Analysis)
* RFE (Recursive Feature Elimination)
* Autoencoder-based reduction

## 🤖 Models Used

### 💡 Machine Learning Models

* Linear Regression
* logistic Regression
* lasso Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor
* Support Vector Regressor (SVR)
* K-Nearest Neighbors (KNN)
* Gradient Boosting

### 🧠 Deep Learning Models

* Feedforward Neural Networks
* Autoencoder + Regression Head
* DNN
* CNN
* Transformer

## 📈 Performance Evaluation

* **Metrics:**

  * R² Score
  * Mean Absolute Error (MAE)
  * Mean Squared Error (MSE)
  * RMSE (Root Mean Squared Error)
* **Comparison Plots:**

  * R² and MAE comparisons between models
  * Actual vs Predicted plots
  * Residual analysis

## 🧪 Results Summary

| Model             | Molar Volume R² | Cohesive Energy R² | MAE |
| ----------------- | --------------- | ------------------ | --- |
| XGBoost Regressor | 0.99            | 0.97               | Low |
| Random Forest     | 0.98            | 0.95               | Low |
| Neural Network    | 0.97            | 0.96               | Low |

> **Best Model:** XGBoost for both properties

## 📁 Project Structure

```
thermodynamic-properties/
│
├── data/                   # Raw and processed datasets
├── descriptors/            # Descriptor generation scripts
├── feature_selection/      # Feature selection scripts and results
├── models/                 # Saved ML and DL models
├── results/                # Evaluation plots and tables
├── notebooks/              # EDA, training, and evaluation notebooks
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## 🛠️ Requirements

* Python ≥ 3.8
* pandas, numpy, scikit-learn, xgboost, tensorflow, keras, matplotlib, seaborn, mordred

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📌 Future Work

* Incorporate temperature/pressure dependence into predictions
* Use transfer learning with pre-trained chemical embeddings
* Deploy model via web interface or API



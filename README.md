# 🏠 Ames Housing Price Prediction

## A Complete Machine Learning Journey From Scratch

---

**Bogdan Banra** | Arad, Arad, Romania

---

## 📋 Overview

This project is a comprehensive end-to-end machine learning implementation focused on predicting house sale prices using the Ames Housing dataset. Rather than relying on off-the-shelf libraries, every algorithm has been implemented **from scratch** to develop a deep, intuitive understanding of the underlying mathematics and mechanics.

The goal was not just to build a predictive model, but to truly understand what happens under the hood when we call `fit()` and `predict()`.

---

## 🎯 Project Objectives

- Implement core machine learning algorithms without sklearn
- Understand PCA dimensionality reduction through eigendecomposition
- Build regularized regression models (Ridge and Lasso) from first principles
- Apply Maximum Likelihood Estimation to linear regression
- Compare model performance across different complexity levels
- Document all mathematical derivations with handwritten notes

---

## 📐 Mathematical Derivations

All mathematical formulas and step-by-step derivations used in this project are documented with handwritten notes.

- [PCA](https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing/tree/main/math/PCA)
- [Lasso](https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing/tree/main/math/Lasso_Regressionn)
- [Linear Regression and MLE](https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing/tree/main/math/Linear_Regression%2BMLE)
- [Ridge Regression and R-Squared](https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing/tree/main/math/Ridge_Regression%2BR_Squared)
- [Standard Scaler](https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing/tree/main/math/Standard_scaler)

---

## 🔧 Implemented From Scratch

### 1. Train-Test Split
- Custom implementation with reproducible shuffling
- Fixed random seed for consistent results
- Handles both DataFrames and NumPy arrays

### 2. Principal Component Analysis (PCA)
- Covariance matrix computation
- Eigendecomposition using `np.linalg.eigh`
- Variance thresholding (tested at 80%, 85%, 90%, 95%)
- Automatic component selection based on explained variance

### 3. Standard Scaler
- Mean centering and standard deviation scaling
- Epsilon safety for zero-variance features
- Separate fit and transform methods

### 4. Linear Regression with MLE
- Ordinary Least Squares via normal equation
- Maximum Likelihood Estimation framework
- Log-likelihood calculation
- AIC and BIC for model comparison
- R-squared and MSE metrics

### 5. Ridge Regression
- L2 regularization penalty
- Closed-form solution with identity matrix
- Lambda hyperparameter tuning

### 6. Lasso Regression
- L1 regularization penalty
- Proximal gradient descent implementation
- Soft-thresholding operator
- Automatic feature selection
- Sparsity tracking

### 7. Polynomial Feature Expansion
- Quadratic feature generation
- Interaction terms between all features

---

## 📊 Dataset

**Ames Housing Dataset** from Kaggle

- **Samples:** 2,930 houses
- **Features:** 80 original columns (expanded to 196 after one-hot encoding)
- **Target:** SalePrice (continuous, USD)

### Feature Categories:
- Lot size and frontage
- Basement quality and square footage
- Garage type, year built, and capacity
- Overall quality and condition ratings
- Neighborhood and zoning classifications

---

## 🔄 Pipeline Overview

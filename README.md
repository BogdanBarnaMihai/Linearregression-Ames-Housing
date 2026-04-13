# 🏠 Ames Housing Price Prediction

## A Complete Machine Learning Journey From Scratch

---

**Bogdan Barna** | Arad, Arad, Romania

---

## 📐 Mathematical Derivations

All mathematical formulas and step-by-step derivations used in this project are documented with handwritten notes.

- [PCA](https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing/tree/main/math/PCA) — Eigendecomposition and variance explained
- [Lasso](https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing/tree/main/math/Lasso_Regressionn) — Soft-threshold operator and proximal gradient
- [Linear Regression and MLE](https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing/tree/main/math/Linear_Regression%2BMLE) — Log-likelihood, AIC, and BIC
- [Ridge Regression and R-Squared](https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing/tree/main/math/Ridge_Regression%2BR_Squared) — L2 penalty and normal equation
- [Standard Scaler](https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing/tree/main/math/Standard_scaler) — Mean centering and variance scaling

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
Raw Data
↓
Data Cleaning (Handling Nulls)
↓
One-Hot Encoding (Categorical → Numeric)
↓
Train-Test Split (80% / 20%)
↓
PCA Dimensionality Reduction (95% variance → 196 components)
↓
Model Training & Hyperparameter Tuning
↓
Model Evaluation & Comparison
---

## 🏆 Key Findings

**Best Model:** OLS (Linear)

**Test RMSE:** $15,827

| Model | Test RMSE | Notes |
|-------|-----------|-------|
| OLS (Linear) | $15,827 | WINNER - Simple, effective, no overfitting |
| Lasso (Linear) | $25,790 | 316 non-zero features, sparsity achieved |
| Lasso (Poly deg=2) | $27,712 | Feature selection with polynomial terms |
| OLS (Poly deg=2) | $36,581 | Slight overfitting with quadratic features |
| Ridge (Poly deg=2) | $181,693 | Excessive regularization |
| Ridge (Linear) | $182,615 | Heavy shrinkage degraded performance |

### Interpretation

The **Ordinary Least Squares linear model** achieved the best performance with a test RMSE of $15,827 — meaning predictions are typically off by about $15,800 on average.

Given that house prices in the dataset range from approximately $50,000 to over $700,000 (mean: $180,796, std: $79,887), this represents an **RMSE-to-Mean ratio of just 8.8%** , indicating strong predictive accuracy.

Interestingly, the more complex models (polynomial features, Ridge regularization, Lasso sparsity) all **underperformed** relative to simple OLS, suggesting that the linear relationships in the data are already well-captured without additional complexity. Lasso's feature selection identified 316 relevant features from the PCA-transformed space, demonstrating its utility for interpretability even when raw predictive power lags behind OLS.

---

## 📈 Visualizations

The notebook includes comprehensive visualizations:

- **Actual vs. Predicted Scatter Plots** — For all 6 model variants
- **Residual Analysis** — Checking for patterns and heteroscedasticity
- **Feature Importance** — Top 15 features selected by Lasso

---

## 🗂️ Repository Structure

```
Linearregression-Ames-Housing/
│
├── README.md
├── Untitled1.ipynb
├── sample_data/
│   └── AmesHousing.csv
│
├── math/
│   ├── PCA/
│   │   └── (PCA eigendecomposition photos)
│   ├── Lasso_Regressionn/
│   │   └── (Lasso soft-threshold photos)
│   ├── Linear_Regression+MLE/
│   │   └── (MLE log-likelihood photos)
│   ├── Ridge_Regression+R_Squared/
│   │   └── (Ridge normal equation photos)
│   └── Standard_scaler/
│       └── (Standardization formula photos)
│
└── images/
    └── (Generated plots and figures)
```

---

## 🚀 How to Run

### Option 1: Google Colab
1. Open the notebook in Google Colab
2. Run all cells sequentially
3. The dataset is loaded from the Colab `sample_data` folder

### Option 2: Local Jupyter Notebook
# Clone the repository
git clone https://github.com/BogdanBarnaMihai/Linearregression-Ames-Housing.git
cd Linearregression-Ames-Housing

# Install requirements
pip install numpy pandas matplotlib scikit-learn

# Launch Jupyter
jupyter notebook Untitled1.ipynb

Note: Update the dataset path to your local AmesHousing.csv location if running locally.

📚 Dependencies

Python 3.x
NumPy
Pandas
Matplotlib
scikit-learn (used only for OneHotEncoder — everything else is from scratch)
💡 Key Learnings

PCA from Scratch: Understanding eigendecomposition reveals how PCA actually rotates the data to maximize variance along new axes.
Lasso vs. Ridge: Lasso's L1 penalty creates sparsity through the soft-thresholding operator, while Ridge's L2 penalty only shrinks coefficients toward zero without eliminating them.
MLE Framework: Linear regression can be viewed as finding parameters that maximize the likelihood of observing the data under Gaussian error assumptions — a powerful probabilistic perspective.
Overfitting with Polynomials: Adding quadratic features doubled the feature count (196 → 392) and increased test error, showing that complexity doesn't always improve performance.
The Power of Simplicity: Simple OLS outperformed all regularized and polynomial models, reminding us that sometimes the simplest approach is best.
Hyperparameter Tuning Matters: Lambda selection dramatically affects Ridge and Lasso performance — too much regularization can destroy predictive power.
🔮 Future Improvements

Implement cross-validation for more robust hyperparameter tuning
Add Elastic Net (combination of L1 and L2 penalties)
Explore feature engineering beyond polynomial expansion
Implement k-fold validation from scratch
Add more advanced optimization algorithms (Adam, RMSprop)
Create a Streamlit web app for interactive predictions
Add unit tests for custom implementations
📝 License

This project is created for educational and portfolio purposes. Feel free to use, modify, and learn from it.

🙏 Acknowledgments

Dataset: Ames Housing dataset from Kaggle
Inspiration: The desire to truly understand machine learning beyond import sklearn
Bogdan Barna | Arad, Arad, Romania


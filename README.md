# house-price-prediction
# 🏠 House Price Prediction Web App

This project is a complete end-to-end machine learning solution for predicting house prices based on key property features. It covers everything from data preprocessing, model building, performance comparison, and feature importance to deploying the final model as an interactive **Streamlit** web app.

---

## 📊 Project Highlights

- ✅ Data cleaning and outlier handling
- 🧠 Feature engineering (log transform, scaling, encoding)
- 🔬 Model experimentation (Linear, Polynomial, SVR, Tree-based, and XGBoost)
- 🏆 Final model: **XGBoost Regressor**
- 🌐 Deployment using **Streamlit** with user input and visualization

---

## 📁 Project Structure


---

## 🧠 ML Workflow Overview

### 🔹 Step 1: Exploratory Data Analysis & Cleaning

- Removed outliers using the IQR method
- Replaced extreme values in columns (bedrooms, bathrooms, stories, parking) with their 75th percentile (Q3)
- Visualized distributions and boxplots (before and after outlier treatment)

### 🔹 Step 2: Feature Engineering

- Log-transformed `price` and `area` → used as `log_area`
- Categorical encoding:
  - Binary columns (e.g., `mainroad`, `airconditioning`) mapped to 0/1
  - `furnishingstatus` ordinally encoded
  - Final one-hot encoding for model input
- Applied **RobustScaler** to numeric features for resistance to outliers

### 🔹 Step 3: Model Building and Comparison

Models Trained:
- Linear Regression
- Polynomial Regression
- Lasso Regression (with CV)
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Regressor (SVR)
- ✅ **XGBoost Regressor** (best performance)

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- R² Score
- Feature Importance (via XGBoost)
- Actual vs Predicted scatter plot

---

## 📊 Final Results

- XGBoost achieved the best **R² score and RMSE**
- Visualization of feature importance showed `log_area`, `airconditioning`, and `prefarea` as top predictors
- Model output was inverse log-transformed using `np.expm1` to return price in actual scale

---

## 🌐 Streamlit Web App

The deployed app allows users to:
- Enter house attributes via sliders and dropdowns
- Instantly predict the price using the XGBoost model
- Visualize model internals (e.g., feature importance)

---

## ▶️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/house_price_app.git
cd house_price_app

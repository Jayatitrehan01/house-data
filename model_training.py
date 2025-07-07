
# House Price Prediction - Full Modeling Code (Improved Version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load dataset
house_data=pd.read_csv("Housing - Housing.csv.csv")

# Boxplot BEFORE outlier handling
numeric_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
for col in numeric_cols:
    sns.boxplot(x=house_data[col])
    plt.title(f'Boxplot BEFORE - {col}')
    plt.show()

# Binary and ordinal encoding
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    house_data[col] = house_data[col].str.strip().str.lower().map({'yes': 1, 'no': 0}).astype('category')

house_data['furnishingstatus'] = house_data['furnishingstatus'].str.strip().str.lower().map({
    'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2
}).astype('category')

# Log transform
house_data['price'] = np.log1p(house_data['price'])
house_data['log_area'] = np.log1p(house_data['area'])
house_data.drop('area', axis=1, inplace=True)

# Replace outliers (except price) with Q3
def replace_outliers_with_q3(data, column):
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - data[column].quantile(0.25)
    lower, upper = Q3 - 1.5 * IQR, Q3 + 1.5 * IQR
    data[column] = np.where((data[column] < lower) | (data[column] > upper), Q3, data[column])
    return data

for col in ['bedrooms', 'bathrooms', 'stories', 'parking']:
    house_data = replace_outliers_with_q3(house_data, col)

# Boxplot AFTER outlier replacement
for col in ['log_area', 'bedrooms', 'bathrooms', 'stories', 'parking']:
    sns.boxplot(x=house_data[col])
    plt.title(f'Boxplot AFTER - {col}')
    plt.show()

# Drop price outliers
Q1, Q3 = house_data['price'].quantile([0.25, 0.75])
IQR = Q3 - Q1
house_data = house_data[(house_data['price'] >= Q1 - 1.5 * IQR) & (house_data['price'] <= Q3 + 1.5 * IQR)]

# Numeric columns
numeric_cols = ['price', 'log_area', 'bedrooms', 'bathrooms', 'stories', 'parking']
house_data[numeric_cols].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(house_data[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Feature scaling
scale_cols = ['log_area', 'bedrooms', 'bathrooms', 'stories', 'parking']
scaler = RobustScaler()
house_data[scale_cols] = scaler.fit_transform(house_data[scale_cols])

# One-hot encode categorical
X = pd.get_dummies(house_data.drop('price', axis=1), drop_first=True)
y = house_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial LR": make_pipeline(PolynomialFeatures(2), LinearRegression()),
    "Lasso Regression": LassoCV(cv=5),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "SVR": SVR(),
    "XGBoost": XGBRegressor()
}

# Train & evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'RMSE': rmse, 'R2 Score': r2})
    print(f"{name}: RMSE={rmse:.2f}, R2={r2:.4f}")

# Save best model and scaler
best_model = models['XGBoost']
import joblib
joblib.dump(best_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

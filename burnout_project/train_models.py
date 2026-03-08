import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib

# Ensure models folder exists
if not os.path.exists("models"):
    os.makedirs("models")

# Load dataset
df = pd.read_csv("dataset/work_from_home_burnout_dataset.csv")

# Features
features = ['work_hours', 'screen_time_hours', 'meetings_count', 'breaks_taken', 
            'after_hours_work', 'sleep_hours', 'task_completion_rate']

X = df[features]

# Target - Continuous
y_score = df['burnout_score']

# Target - Classification
y_risk = df['burnout_risk']
le = LabelEncoder()
y_risk_encoded = le.fit_transform(y_risk)  # Low/Medium/High → 0/1/2

# Split for training (optional)
X_train, X_test, y_train_score, y_test_score = train_test_split(X, y_score, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_risk_encoded, test_size=0.2, random_state=42)

# ----- Regression models for burnout_score -----
# Decision Tree Regressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train_score)
joblib.dump(dt_model, "models/decision_tree_model.pkl")

# Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train_score)
joblib.dump(rf_model, "models/random_forest_model.pkl")

# KNN Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train_score)
joblib.dump(knn_model, "models/knn_model.pkl")

# Linear Regression
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train_score)
joblib.dump(mlr_model, "models/mlr_model.pkl")

# ----- Classification model for burnout_risk -----
from sklearn.tree import DecisionTreeClassifier
dtc_model = DecisionTreeClassifier()
dtc_model.fit(X_train_r, y_train_r)
joblib.dump(dtc_model, "models/decision_tree_risk_model.pkl")

print("All models trained and saved successfully!")
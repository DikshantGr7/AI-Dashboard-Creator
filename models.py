import pandas as pd
import numpy as np # Needed for sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import time

# --- Configuration ---
DATA_FILE = "amazon_sales_2025_INR.csv"
MODELS_DIR = "trained_models"
MODEL_FILES = {
    'LinearRegression': os.path.join(MODELS_DIR, 'linear_regression_predictor.pkl'),
    'RandomForest': os.path.join(MODELS_DIR, 'random_forest_predictor.pkl'),
    'XGBoost': os.path.join(MODELS_DIR, 'xgboost_predictor.pkl'),
}

def train_and_save_model(model_name, model_estimator, X_train, X_test, Y_train, Y_test, preprocessor):
    """Creates a pipeline, trains the model, evaluates it (including RMSE), and saves the pipeline."""
    
    start_time = time.time()
    
    # 1. Create Full Pipeline (Preprocessor + Estimator)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model_estimator)])
    
    # 2. Train
    print(f"\n--- Training {model_name}...")
    pipeline.fit(X_train, Y_train)
    
    # 3. Evaluate
    Y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse) # Calculate RMSE
    r2 = r2_score(Y_test, Y_pred)
    
    end_time = time.time()
    
    print(f"   Training Time: {end_time - start_time:.2f} seconds")
    print(f"   Mean Squared Error (MSE): {mse:,.2f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"   R-squared (R2): {r2:.4f}")

    # 4. Save
    filepath = MODEL_FILES[model_name]
    joblib.dump(pipeline, filepath)
    print(f"   Model saved to '{filepath}'")
    
    return {'model': model_name, 'r2': r2, 'mse': mse, 'rmse': rmse, 'time': end_time - start_time}


def train_sales_predictors():
    """Trains Linear Regression, Random Forest, and XGBoost models."""
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    print(f"Starting model training for {DATA_FILE}...")
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at '{DATA_FILE}'. Please ensure it is in the same directory.")
        return
        
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- Data Preparation ---
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    
    Y = df['Total_Sales_INR']
    X = df[['Day_of_Year', 'Quantity', 'Unit_Price_INR', 'Product_Category']]
    
    # Clean data (essential for robust training)
    X = X.dropna()
    Y = Y[X.index] 
    print(f"Dataset size after cleaning: {len(X)} rows.")

    # --- Preprocessor Setup ---
    numerical_features = ['Day_of_Year', 'Quantity', 'Unit_Price_INR']
    categorical_features = ['Product_Category']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # --- Split Data ---
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # --- Define Models to Train ---
    models_to_train = [
        ('LinearRegression', LinearRegression()), 
        ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)), 
        ('XGBoost', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)), 
    ]
    
    # --- Train All Models ---
    results = []
    for name, estimator in models_to_train:
        result = train_and_save_model(name, estimator, X_train, X_test, Y_train, Y_test, preprocessor)
        results.append(result)

    print("\n===================================")
    print("      ALL MODELS TRAINING COMPLETE")
    print("===================================")
    for res in results:
        print(f"Model: {res['model']} | R2: {res['r2']:.4f} | RMSE: {res['rmse']:,.2f} | Time: {res['time']:.2f}s")
    print("===================================")


if __name__ == '__main__':
    train_sales_predictors()
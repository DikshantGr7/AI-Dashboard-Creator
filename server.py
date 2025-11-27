import uvicorn
import pandas as pd
import random
import json
import joblib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Literal
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration ---
MODELS_DIR = "trained_models"
MODEL_FILES = {
    'LinearRegression': os.path.join(MODELS_DIR, 'linear_regression_predictor.pkl'),
    'RandomForest': os.path.join(MODELS_DIR, 'random_forest_predictor.pkl'),
    'XGBoost': os.path.join(MODELS_DIR, 'xgboost_predictor.pkl'),
}
MOCK_DATA_STORE = {} 

app = FastAPI(title="PyDash Analytics ML API")

# --- Model Loading ---
SALES_PREDICTORS: Dict[str, Pipeline] = {} 
AVAILABLE_MODELS = list(MODEL_FILES.keys())

# Load all models at application startup
for model_name, path in MODEL_FILES.items():
    try:
        if os.path.exists(path):
            SALES_PREDICTORS[model_name] = joblib.load(path)
            print(f"✓ Successfully loaded {model_name} model from {path}")
        else:
            print(f"⚠ Warning: {model_name} model not found at '{path}'. Please run models.py first.")
    except Exception as e:
        print(f"✗ Error loading {model_name} model: {e}")

# --- CORS Configuration ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Models ---

class LoginRequest(BaseModel):
    email: str
    password: str

class ChartRequest(BaseModel):
    user_id: str
    filename: str
    chart_type: str
    algo_type: str
    x_axis: str
    y_axis: str
    title: str

class AiRequest(BaseModel):
    user_id: str
    filename: str
    context: str

class PredictionRequest(BaseModel):
    model_name: Literal['LinearRegression', 'RandomForest', 'XGBoost']
    product_category: str
    quantity: int
    unit_price_inr: float
    
# --- Helper Functions ---

def preprocess_for_prediction(data: Dict[str, Any]) -> pd.DataFrame:
    """Converts request data into a DataFrame format expected by the model pipeline."""
    now = datetime.now()
    day_of_year = now.timetuple().tm_yday
    
    data_for_df = {
        'Day_of_Year': [day_of_year], 
        'Quantity': [data['quantity']],
        'Unit_Price_INR': [data['unit_price_inr']],
        'Product_Category': [data['product_category']]
    }
    
    return pd.DataFrame(data_for_df, columns=['Day_of_Year', 'Quantity', 'Unit_Price_INR', 'Product_Category'])

def aggregate_data(df: pd.DataFrame, x_col: str, y_col: str, agg_type: str):
    """Aggregates dataframe based on x column and aggregation type."""
    if agg_type == 'sum':
        return df.groupby(x_col)[y_col].sum().reset_index()
    elif agg_type == 'mean':
        return df.groupby(x_col)[y_col].mean().reset_index()
    elif agg_type == 'count':
        return df.groupby(x_col)[y_col].count().reset_index()
    return df

# --- Endpoints ---

@app.post("/login")
async def login(req: LoginRequest):
    """Mocks user login and returns a user ID."""
    if req.email == "demo@user.com" and req.password == "password":
        return {"message": "Login successful", "user_id": "mock_user_123"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/upload")
async def upload_file(user_id: str = Form(...), file: UploadFile = File(...)):
    """Processes uploaded file and stores data."""
    content = await file.read()
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        # Store the actual dataframe
        MOCK_DATA_STORE[user_id] = df 
        columns = df.columns.tolist()
        rows = len(df)
        
        print(f"✓ Uploaded dataset for {user_id}: {rows} rows, {len(columns)} columns")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing failed: {e}")

    return {
        "filename": f"data_{user_id}", 
        "rows": rows,
        "columns": columns,
        "message": "File processed successfully"
    }

@app.post("/create_chart")
async def create_chart(req: ChartRequest):
    """Creates chart from actual uploaded data."""
    
    # Check if user data exists
    if req.user_id not in MOCK_DATA_STORE:
        raise HTTPException(status_code=404, detail="No data found. Please upload a file first.")
    
    df = MOCK_DATA_STORE[req.user_id]
    
    # Validate columns
    if req.x_axis not in df.columns or req.y_axis not in df.columns:
        raise HTTPException(status_code=400, detail=f"Columns {req.x_axis} or {req.y_axis} not found in dataset")
    
    try:
        # Aggregate data
        agg_df = aggregate_data(df, req.x_axis, req.y_axis, req.algo_type)
        
        x_data = agg_df[req.x_axis].tolist()
        y_data = agg_df[req.y_axis].tolist()
        
        # Determine y-axis label
        if req.algo_type == 'sum':
            y_label = f"Total {req.y_axis}"
        elif req.algo_type == 'mean':
            y_label = f"Average {req.y_axis}"
        else:
            y_label = f"Count of {req.y_axis}"
        
        # Create chart based on type
        if req.chart_type == 'pie':
            graph_json = {
                "data": [{
                    "labels": x_data,
                    "values": y_data,
                    "type": "pie",
                    "marker": {"colors": px.colors.qualitative.Set3}
                }],
                "layout": {
                    "title": req.title,
                    "template": "plotly_white",
                    "height": 320
                }
            }
        else:
            graph_json = {
                "data": [{
                    "x": x_data,
                    "y": y_data,
                    "name": req.title,
                    "type": req.chart_type,
                    "marker": {"color": "#3b82f6"}
                }],
                "layout": {
                    "title": req.title,
                    "xaxis": {"title": req.x_axis},
                    "yaxis": {"title": y_label},
                    "template": "plotly_white",
                    "height": 320
                }
            }
        
        return {"graph_json": json.dumps(graph_json)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart creation failed: {e}")

@app.post("/get_ai_feedback")
async def get_ai_feedback(req: AiRequest):
    """Provides AI feedback on data (mock implementation)."""
    
    if req.user_id not in MOCK_DATA_STORE:
        raise HTTPException(status_code=404, detail="No data found for analysis")
    
    df = MOCK_DATA_STORE[req.user_id]
    
    # Generate basic statistics for context
    stats = f"Dataset contains {len(df)} rows and {len(df.columns)} columns."
    
    mock_insights = [
        f"Based on your data: {stats} The distribution appears normal with some outliers in the upper quartile.",
        f"Analysis shows: {stats} Consider focusing on the top 20% of categories which drive 80% of value.",
        f"Data quality check: {stats} No significant missing values detected. Dataset is clean for modeling.",
        f"Trend analysis: {stats} There's a positive correlation between quantity and sales value as expected."
    ]
    
    feedback = random.choice(mock_insights)
    
    if req.context:
        feedback += f"\n\nRegarding your question: '{req.context}' - I recommend examining the seasonal patterns and category performance metrics."
    
    return {"feedback": feedback}

@app.post("/predict_sales")
async def predict_sales(req: PredictionRequest):
    """Uses trained model to predict Total_Sales_INR."""
    model_name = req.model_name
    predictor = SALES_PREDICTORS.get(model_name)
    
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model '{model_name}' not loaded. Run 'python models.py' first to train models."
        )

    try:
        input_df = preprocess_for_prediction(req.model_dump())
        prediction = predictor.predict(input_df)[0]
        
        return {
            "model_used": model_name,
            "prediction_inr": max(0, round(float(prediction), 2)),
            "message": f"Sales prediction successful using {model_name}."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}. Verify input features match training data."
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(SALES_PREDICTORS.keys()),
        "models_available": AVAILABLE_MODELS
    }

# --- Runner ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("    PyDash Backend Starting")
    print("="*50)
    print(f"FastAPI Server: http://localhost:8000")
    print(f"Available Models: {AVAILABLE_MODELS}")
    print(f"Models Loaded: {list(SALES_PREDICTORS.keys())}")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
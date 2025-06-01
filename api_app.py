from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
# Load preprocessor pipeline for input column processing to make them compatible to pass through the model
preprocessor = joblib.load('preprocessor.joblib')
# Load models
dt_model = joblib.load('DecisionTree_model.joblib')
rf_model = joblib.load('RandomForest_model.joblib')
xgb_model = joblib.load('XGBoost_model.joblib')

# Model mapping
model_dict = {
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

# Mapping from label to integer
target_map_dict = {
    'functional': 2,
    'functional needs repair': 1,
    'non functional': 0
}

# Inverse mapping: from integer back to label
inv_target_map_dict = {v: k for k, v in target_map_dict.items()}

app= FastAPI()
class PredictionRequest(BaseModel):
    df: List[Dict[str, Any]]   
    model_name: str



@app.post("/make_prediction")
def MAKE_PREDICTION(request: PredictionRequest): # under the hood data is parsed using request=PredictionRequest(**request.json()).Now request is a pydentic object with type hint PredictionRequest
    try:
        input_df = pd.DataFrame(request.df)
        model_name = request.model_name

        if model_name not in model_dict:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found.")

        model = model_dict[model_name]

        # Preprocess first
        transformed_input = preprocessor.transform(input_df)

        # Then predict
        prediction = model.predict(transformed_input)[0]
        label_prediction = inv_target_map_dict.get(prediction, "Unknown")

        return {
            "prediction": label_prediction,
            "Code": int(prediction)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/make_prediction_testdata")
def make_prediction_testdata(request: PredictionRequest):
    try:
        input_df = pd.DataFrame(request.df).replace({None: np.nan})
        model_name = request.model_name

        if model_name not in model_dict:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found.")

        if 'target' not in input_df.columns:
            raise HTTPException(status_code=400, detail="Missing 'target' column for accuracy calculation.")

        # Separate target and features
        y_true = input_df['target']
      

        X = input_df.drop(columns=['target'])

        # Transform input and predict
        X_transformed = preprocessor.transform(X)
        model = model_dict[model_name]
        y_pred = model.predict(X_transformed)

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)


        return {
            "accuracy": accuracy
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

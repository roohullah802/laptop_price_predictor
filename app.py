from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd


with open("pipe.pkl", "rb") as f:
    pipeline = pickle.load(f)
with open("x_train.pkl", "rb") as d:
    X_train = pickle.load(d)    

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


class PredictionInput(BaseModel):
    Company: str
    TypeName: str
    Weight: float
    cpu_brand: str
    Ram: int
    gpu: str
    os: str
    Touch_screen: str
    IPS_Panel: str
    screenSize: float
    screenResolution: str
    ssd: int
    hdd: int

@app.get("/")
def home():
    if X_train is None:
        return {"error": "Model or training data not loaded. Check the files."}

    try:
        # Extract feature names
        preprocessor = pipeline.named_steps["step1"]  # Get preprocessor step
        numerical_features = preprocessor.transformers_[0][2]  # Numerical columns
        categorical_features = preprocessor.transformers_[1][1].get_feature_names_out().tolist()  # OHE categorical columns

        feature_names = numerical_features + categorical_features  # Combine all features

        # Extract distinct values for each feature
        feature_values = {col: X_train[col].unique().tolist() for col in feature_names}

        return {
            "model_features": feature_names,
            "feature_values": feature_values  # Dictionary of feature-wise unique values
        }

    except Exception as e:
        return {"error": str(e)}

    


@app.post("/predict")
def predict_price(data: PredictionInput):
   
    try:
        data.Touch_screen = 1 if data.Touch_screen == "Yes" else 0
        data.IPS_Panel = 1 if data.IPS_Panel == "Yes" else 0
    
        x_res, y_res = map(int, data.screenResolution.split("x"))
        ppi = ((x_res**2 + y_res**2) ** 0.5) / data.screenSize


  
        input_df = pd.DataFrame([{
            "Company": data.Company,
            "TypeName": data.TypeName,
            "Weight": data.Weight,
            "Cpu brand": data.cpu_brand, 
            "Ram": data.Ram,
            "Gpu brand": data.gpu, 
            "os": data.os,
            "Touch_screen": data.Touch_screen,
            "IPS Panel": data.IPS_Panel,  
            "ppi": ppi,  
            "SSD (GB)": data.ssd,  
            "HDD (GB)": data.hdd  
        }]) 

      
        transformed_input = pipeline.named_steps["step1"].transform(input_df)
       

       
        prediction = pipeline.named_steps["step2"].predict(transformed_input)[0]

        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        return {"error": str(e)}

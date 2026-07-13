from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

bundle = joblib.load("fraud_model_bundle.pkl")

model = bundle["model"]
encoder = bundle["encoder"]
categorical_cols = bundle["categorical_cols"]
numeric_cols = bundle["numeric_cols"]

app = FastAPI(title="Credit Card Fraud Detection API")


class Transaction(BaseModel):
    merchant: str
    category: str
    amt: float
    city: str
    state: str
    city_pop: int
    trans_hour: int
    day_of_week: int


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):

    input_data = pd.DataFrame([transaction.dict()])
    input_data["is_night"] = (input_data["trans_hour"] < 6).astype(int)
    input_data["is_late_night"] = (input_data["trans_hour"] > 22).astype(int)
    input_data["high_amount_flag"] = (input_data["amt"] > 200).astype(int)

    encoded_cat = encoder.transform(input_data[categorical_cols])
    encoded_cat = pd.DataFrame(encoded_cat, columns=categorical_cols)

    final_input = pd.concat(
        [encoded_cat.reset_index(drop=True),
         input_data[numeric_cols].reset_index(drop=True)],
        axis=1
    )

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "prediction": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": round(float(probability), 4),
        "risk_level": risk_level
    }
from fastapi import FastAPI
import joblib, json
import numpy as np
import pandas as pd
from pydantic import BaseModel

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()

# ---------------------------
# Load model & features
# ---------------------------
pipeline = joblib.load("panic_model.pkl")
feature_names = json.load(open("feature_names.json"))

WINDOW_SIZE = 20
THRESHOLD = 0.7
buffer = []
counter = 0
alarm_on = False
ALARM_HOLD = 5  # عدد القراءات اللي نحتفظ فيها بالalarm True

# ---------------------------
# Sensor Data Model
# ---------------------------
class SensorData(BaseModel):
    heart_rate: float
    hrv: float
    gsr_value: float
    temperature: float
    ax: float
    ay: float
    az: float

# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(window):
    feats = {}
    cols = ["HR", "HRV", "EDA", "TEMP", "ACC_X", "ACC_Y", "ACC_Z"]
    df = pd.DataFrame(window, columns=cols)
    for col in cols:
        sig = df[col].values
        feats[f"{col}_mean"]   = np.mean(sig)
        feats[f"{col}_std"]    = np.std(sig)
        feats[f"{col}_min"]    = np.min(sig)
        feats[f"{col}_max"]    = np.max(sig)
        feats[f"{col}_range"]  = np.max(sig) - np.min(sig)
        feats[f"{col}_median"] = np.median(sig)
    return feats

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
def predict(data: SensorData):
    global buffer, counter, alarm_on

    # Append new reading
    row = [data.heart_rate, data.hrv, data.gsr_value, 
           data.temperature, data.ax, data.ay, data.az]
    buffer.append(row)

    # Maintain sliding window size
    if len(buffer) > WINDOW_SIZE:
        buffer.pop(0)

    # Return collecting status if not enough data
    if len(buffer) < WINDOW_SIZE:
        return {
            "status": "collecting data",
            "buffer_size": len(buffer)
        }

    # Extract features from window
    feats = extract_features(buffer)
    df_feats = pd.DataFrame([feats])[feature_names]

    # Predict
    prob = pipeline.predict_proba(df_feats)[0][1]
    pred = int(prob > THRESHOLD)

    # Update counter for alarm stability
    if pred == 1:  # High Risk
        counter += 1
    else:          # Low Risk
        counter = 0
        alarm_on = False

    alarm = False
    if counter >= 3 and not alarm_on:
        alarm = True
        alarm_on = True
    elif alarm_on:
        # Keep alarm on for ALARM_HOLD readings
        if counter > 0:
            alarm = True
        else:
            alarm_on = False

    return {
        "risk": "High Risk" if pred == 1 else "Low Risk",
        "confidence": float(round(prob * 100, 2)),
        "alarm": bool(alarm),
        "buffer_size": int(len(buffer))
    }
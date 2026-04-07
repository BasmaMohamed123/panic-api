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
pipeline = joblib.load(r"C:\Users\admin\Downloads\graduation project\panic_model.pkl")
feature_names = json.load(open(r"C:\Users\admin\Downloads\graduation project\feature_names.json"))

WINDOW_SIZE = 20
THRESHOLD = 0.7
buffer = []
counter = 0
alarm_on = False

class SensorData(BaseModel):
    heart_rate: float
    hrv: float
    gsr_value: float
    temperature: float
    ax: float
    ay: float
    az: float

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

@app.post("/predict")
def predict(data: SensorData):
    global buffer, counter, alarm_on

    row = [data.heart_rate, data.hrv, data.gsr_value, 
           data.temperature, data.ax, data.ay, data.az]
    
    buffer.append(row)

    if len(buffer) < WINDOW_SIZE:
        return {
            "status": "collecting data",
            "buffer_size": len(buffer)
        }

    window = buffer[-WINDOW_SIZE:]
    feats = extract_features(window)
    df_feats = pd.DataFrame([feats])[feature_names]

    prob = pipeline.predict_proba(df_feats)[0][1]
    pred = int(prob > THRESHOLD)

    if prob > THRESHOLD:
        counter += 1
    else:
        counter = 0
        alarm_on = False

    alarm = False
    if counter >= 3 and not alarm_on:
        alarm = True
        alarm_on = True

    return {
            "risk": "High Risk" if pred == 1 else "Low Risk",
            "confidence": float(round(prob * 100, 2)),  # float بدلاً من numpy.float32
            "alarm": bool(alarm),                        # bool بدلاً من numpy.bool_
            "buffer_size": int(len(buffer))              # int بدلاً من numpy.int64
        }

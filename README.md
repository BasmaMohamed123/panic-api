# GuardBand – Panic Risk Detection API

GuardBand is an AI-based wearable system designed to monitor physiological signals and classify panic risk into **Low Risk** or **High Risk**.

This repository contains the **FastAPI backend and machine learning API** used to serve real-time panic-risk predictions.

## Project Overview

The system uses physiological data collected from a wearable device, including:

* Heart Rate (HR)
* Heart Rate Variability (HRV)
* Galvanic Skin Response (GSR)
* Body Temperature
* Accelerometer data (X, Y, Z)

The collected data is processed and passed to a trained machine learning model to predict the current panic-risk level.

## Machine Learning

The ML pipeline includes:

* Data preprocessing
* Feature engineering
* Feature scaling
* Model training
* Model evaluation
* Real-time prediction

The project uses **XGBoost** as the main classification model, with Random Forest used during model evaluation and comparison.

### Prediction Classes

* **Low Risk**
* **High Risk**

## API

The backend is implemented using **FastAPI** and provides an endpoint for making predictions from wearable sensor data.

Example input:

```json
{
  "heart_rate": 95,
  "hrv": 0.12,
  "gsr_value": 4.2,
  "temperature": 36.7,
  "ax": 0.12,
  "ay": -0.05,
  "az": 0.98
}
```

The API returns the predicted risk level along with the model confidence.

Example:

```json
{
  "prediction": "High Risk",
  "confidence": 0.9485
}
```

## Technologies

* Python
* Scikit-learn
* XGBoost
* FastAPI
* Pandas
* NumPy
* Joblib

## Project Structure

```text
Panic.API/
│
├── main.py
├── panic_model.pkl
├── feature_names.json
├── requirements.txt
└── README.md
```

## Deployment

The FastAPI service was deployed as a REST API to allow the mobile application to send sensor data and receive real-time panic-risk predictions.

## Graduation Project

GuardBand was developed as a graduation project at the **Faculty of Engineering, El Shorouk Academy**.

The complete system combines:

**Wearable Device → Mobile Application → AI Model → FastAPI → Risk Prediction**

The wearable device collects physiological signals, while the AI backend processes the data and provides the predicted risk level to the application.

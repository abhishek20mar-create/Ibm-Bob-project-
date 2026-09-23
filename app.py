"""
app.py — FastAPI Backend for Heart Attack Prediction
Endpoints:
  GET  /             — health check
  GET  /model/stats  — model performance metrics + feature importances
  POST /predict      — predict heart attack risk from patient features
"""

import json
import os
import pickle
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from src.predict import _risk_level

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "model")
PIPELINE_PATH  = os.path.join(MODEL_DIR, "pipeline.pkl")
STATS_PATH     = os.path.join(MODEL_DIR, "stats.json")
METADATA_PATH  = os.path.join(MODEL_DIR, "feature_metadata.json")

# ── Load artifacts at startup ────────────────────────────────────────────────
if not os.path.exists(PIPELINE_PATH):
    raise RuntimeError(
        "Model not found. Run `python train.py` first to train and save the model."
    )

with open(PIPELINE_PATH, "rb") as f:
    PIPELINE = pickle.load(f)

with open(STATS_PATH) as f:
    STATS = json.load(f)

with open(METADATA_PATH) as f:
    FEATURE_METADATA = json.load(f)

FEATURE_COLS = STATS["feature_cols"]

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Attack Prediction API",
    description="Predict heart attack risk using a Random Forest model trained on the Cleveland Heart Disease dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class PatientFeatures(BaseModel):
    age:      float = Field(..., ge=1,   le=120, description="Age in years")
    sex:      float = Field(..., ge=0,   le=1,   description="Sex (0=Female, 1=Male)")
    cp:       float = Field(..., ge=1,   le=4,   description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50,  le=250, description="Resting blood pressure (mmHg)")
    chol:     float = Field(..., ge=50,  le=700, description="Serum cholesterol (mg/dl)")
    fbs:      float = Field(..., ge=0,   le=1,   description="Fasting blood sugar >120 mg/dl")
    restecg:  float = Field(..., ge=0,   le=2,   description="Resting ECG results (0-2)")
    thalach:  float = Field(..., ge=50,  le=250, description="Max heart rate achieved")
    exang:    float = Field(..., ge=0,   le=1,   description="Exercise induced angina (0/1)")
    oldpeak:  float = Field(..., ge=0.0, le=10.0,description="ST depression induced by exercise")
    slope:    float = Field(..., ge=1,   le=3,   description="Slope of peak exercise ST segment (1-3)")
    ca:       float = Field(..., ge=0,   le=3,   description="Number of major vessels (0-3)")
    thal:     float = Field(..., ge=0,   le=7,   description="Thalassemia type (3/6/7)")

    class Config:
        schema_extra = {
            "example": {
                "age": 54, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
                "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
                "oldpeak": 1.4, "slope": 2, "ca": 0, "thal": 3
            }
        }


class PredictionResponse(BaseModel):
    prediction:   int
    label:        str
    probability:  float
    risk_level:   str
    top_features: list


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "Heart Attack Prediction API is running"}


@app.get("/model/stats", tags=["Model"])
def get_model_stats():
    """Return model performance metrics, confusion matrix, and feature importances."""
    return STATS


@app.get("/model/features", tags=["Model"])
def get_feature_metadata():
    """Return feature descriptions, ranges, and option labels for the UI."""
    return FEATURE_METADATA


@app.get("/model/comparison", tags=["Model"])
def get_model_comparison():
    """Return per-model metrics for Logistic Regression, Random Forest, and XGBoost."""
    return STATS.get("all_models", [])




@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientFeatures):
    """Predict heart attack risk for a patient."""
    try:
        features = np.array([[
            patient.age, patient.sex, patient.cp, patient.trestbps,
            patient.chol, patient.fbs, patient.restecg, patient.thalach,
            patient.exang, patient.oldpeak, patient.slope, patient.ca, patient.thal
        ]])

        prediction  = int(PIPELINE.predict(features)[0])
        probability = float(PIPELINE.predict_proba(features)[0][1])

        # Risk level — delegate to src.predict to keep thresholds in one place
        risk_level = _risk_level(probability)

        label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        # Top contributing features (from RF impurity importances)
        importances = STATS["feature_importances"]
        top_features = sorted(
            [{"feature": k, "importance": round(v, 4)} for k, v in importances.items()],
            key=lambda x: x["importance"],
            reverse=True,
        )[:5]

        return PredictionResponse(
            prediction=prediction,
            label=label,
            probability=round(probability, 4),
            risk_level=risk_level,
            top_features=top_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

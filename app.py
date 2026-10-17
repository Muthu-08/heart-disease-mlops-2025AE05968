import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# Configure standard structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("heart-disease-api")

# Initialize FastAPI application
app = FastAPI(
    title="Heart Disease Risk API", 
    version="1.0.0",
    description="Production-ready REST API serving a trained clinical ML classifier."
)

# Integrate Prometheus metrics instrumentation (/metrics)
Instrumentator().instrument(app).expose(app)

# Load the serialised trained model pipeline
model = joblib.load("models/best_model.joblib")


class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float

    # Pydantic V2 schema configuration
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 55.0,
                    "sex": 1,
                    "cp": 2,
                    "trestbps": 130.0,
                    "chol": 250.0,
                    "fbs": 0,
                    "restecg": 1,
                    "thalach": 145.0,
                    "exang": 0,
                    "oldpeak": 1.5,
                    "slope": 1,
                    "ca": 0.0,
                    "thal": 3.0
                }
            ]
        }
    }


@app.get("/health")
def health_check():
    """Liveness probe endpoint for Kubernetes / Docker health checks."""
    return {"status": "healthy"}


@app.post("/predict")
def predict_risk(data: PatientData):
    """
    Accepts patient health records and outputs predicted risk class,
    label, and the model's confidence score.
    """
    try:
        # Transform incoming Pydantic schema to pandas DataFrame for sklearn pipeline
        payload = pd.DataFrame([data.model_dump()])
        
        # Inference
        probability = float(model.predict_proba(payload)[0][1])
        prediction = int(probability >= 0.5)
        risk_label = "High" if prediction == 1 else "Low"

        logger.info(f"Processed request | Risk: {risk_label} | Confidence: {probability:.4f}")
        
        return {
            "prediction": prediction,
            "risk_label": risk_label,
            "confidence": round(probability, 4)
        }
    except Exception as exc:
        logger.error(f"Inference error encountered: {str(exc)}")
        raise HTTPException(status_code=500, detail="Inference execution error")
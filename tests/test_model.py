import pytest
import pandas as pd
import numpy as np
import joblib

@pytest.fixture
def sample_payload():
    return {
        "age": 58, "sex": 1, "cp": 2, "trestbps": 120,
        "chol": 240, "fbs": 0, "restecg": 1, "thalach": 160,
        "exang": 0, "oldpeak": 1.2, "slope": 1, "ca": 0.0, "thal": 3.0
    }

def test_inference_pipeline(sample_payload):
    model = joblib.load("models/best_model.joblib")
    df = pd.DataFrame([sample_payload])
    
    pred = model.predict(df)
    prob = model.predict_proba(df)
    
    assert pred[0] in [0, 1]
    assert 0.0 <= prob[0][1] <= 1.0
    assert np.isclose(prob[0].sum(), 1.0)
import pandas as pd


def predict(pipeline, input_data: dict) -> float:
    """Return the survival probability (0–1) for a single company."""
    X = pd.DataFrame([input_data])
    proba = pipeline.predict_proba(X)
    return float(proba[0][1])

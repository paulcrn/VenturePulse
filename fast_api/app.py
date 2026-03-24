import shap
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from project_logic.registry import load_model_trained
from project_logic.shap_translator import explain_prediction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: load pipeline once and cache the SHAP explainer
# ---------------------------------------------------------------------------
_pipeline     = load_model_trained()
_preprocessor = _pipeline.named_steps['prep']
_model        = _pipeline.named_steps['model']
_explainer    = shap.TreeExplainer(_model)

# Build ordered feature names from the fitted preprocessor
_num_features = list(_preprocessor.transformers_[0][2])
_ohe          = _preprocessor.named_transformers_['cat'].named_steps['onehot']
_ohe_names    = _ohe.get_feature_names_out(['region_group', 'industry_group']).tolist()
_feature_names = _num_features + _ohe_names


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return {"status": "ok"}


@app.get("/predict")
def make_prediction(
    funding_total_usd: float,
    funding_rounds: float,
    seed: float,
    venture: float,
    debt_financing: float,
    angel: float,
    grant: float,
    private_equity: float,
    round_A: float,
    round_B: float,
    round_C: float,
    round_D: float,
    avg_raised_per_round: float,
    age_first_funding_days: float,
    funding_span_days: float,
    avg_years_between_rounds: float,
    time_since_last_funding: float,
    region_group: str,
    industry_group: str,
):
    input_data = {
        "funding_total_usd":        funding_total_usd,
        "funding_rounds":           funding_rounds,
        "seed":                     seed,
        "venture":                  venture,
        "debt_financing":           debt_financing,
        "angel":                    angel,
        "grant":                    grant,
        "private_equity":           private_equity,
        "round_A":                  round_A,
        "round_B":                  round_B,
        "round_C":                  round_C,
        "round_D":                  round_D,
        "avg_raised_per_round":     avg_raised_per_round,
        "age_first_funding_days":   age_first_funding_days,
        "funding_span_days":        funding_span_days,
        "avg_years_between_rounds": avg_years_between_rounds,
        "time_since_last_funding":  time_since_last_funding,
        "region_group":             region_group,
        "industry_group":           industry_group,
    }

    X          = pd.DataFrame([input_data])
    probability = float(_pipeline.predict_proba(X)[0][1])

    X_pp        = _preprocessor.transform(X)
    shap_values = _explainer.shap_values(X_pp)

    return explain_prediction(shap_values[0], _feature_names, probability, feature_values=X_pp[0], raw_input=input_data)

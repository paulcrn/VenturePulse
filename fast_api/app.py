import shap
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from project_logic.registry import load_model_trained
from project_logic.shap_translator import explain_prediction

# Median avg_years_between_rounds for single-round companies (from training data)
_AVG_YEARS_SINGLE_ROUND = 1.040383

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
    # Raw date inputs
    founded_at: str,
    first_funding_at: str,
    last_funding_at: str,
    # Funding totals
    funding_total_usd: float,
    funding_rounds: float,
    # Funding type flags (0 or 1)
    seed: float,
    venture: float,
    debt_financing: float,
    angel: float,
    grant: float,
    private_equity: float,
    # Round flags (0 or 1)
    round_A: float,
    round_B: float,
    round_C: float,
    round_D: float,
    # Groupings
    region_group: str,
    industry_group: str,
):
    try:
        t_founded      = pd.Timestamp(founded_at)
        t_first        = pd.Timestamp(first_funding_at)
        t_last         = pd.Timestamp(last_funding_at)
    except Exception:
        raise HTTPException(status_code=422, detail="Dates must be in YYYY-MM-DD format.")

    today = pd.Timestamp.today().normalize()

    # --- Derived features (mirrors training feature engineering) ---
    avg_raised_per_round = (funding_total_usd / funding_rounds) if funding_rounds > 0 else 0.0

    age_first_funding_days = max(0.0, (t_first - t_founded).days)

    funding_span_days = max(0.0, (t_last - t_first).days)

    if funding_rounds > 1:
        avg_years_between_rounds = (funding_span_days / 365.25) / (funding_rounds - 1)
    else:
        avg_years_between_rounds = _AVG_YEARS_SINGLE_ROUND

    # Use today as snapshot so real post-2015 startups land in-distribution
    time_since_last_funding = max(0.0, (today - t_last).days / 365.25)

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

    X           = pd.DataFrame([input_data])
    probability = float(_pipeline.predict_proba(X)[0][1])

    X_pp        = _preprocessor.transform(X)
    shap_values = _explainer.shap_values(X_pp)

    return explain_prediction(shap_values[0], _feature_names, probability, feature_values=X_pp[0], raw_input=input_data)

from fastapi import FastAPI
from pydantic import BaseModel

from project_logic.registry import load_model_trained
from project_logic.predict import predict

app = FastAPI()

app.state.model = load_model_trained()

# # Allow all requests (optional, good for development purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

@app.get("/")
def index():
    return {"status": "ok"}

@app.get("/predict")
def make_prediction(
    avg_raised_per_round: float,
    age_first_funding_days: float,
    has_multiple_rounds: float,
    funding_span_days: float,
    avg_years_between_rounds: float,
    funding_rounds: float,
    funding_total_usd: float,
    seed: float,
    venture: float,
    angel: float,
    grant: float,
    debt_financing: float,
    private_equity: float,
    round_A: float,
    round_B: float,
    round_C: float,
    round_D: float,
    round_E: float,
    region_group: str,
    industry_group: str
):
    input_data = {
        "avg_raised_per_round": avg_raised_per_round,
        "age_first_funding_days": age_first_funding_days,
        "has_multiple_rounds": has_multiple_rounds,
        "funding_span_days": funding_span_days,
        "avg_years_between_rounds": avg_years_between_rounds,
        "funding_rounds": funding_rounds,
        "funding_total_usd": funding_total_usd,
        "seed": seed,
        "venture": venture,
        "angel": angel,
        "grant": grant,
        "debt_financing": debt_financing,
        "private_equity": private_equity,
        "round_A": round_A,
        "round_B": round_B,
        "round_C": round_C,
        "round_D": round_D,
        "round_E": round_E,
        "region_group": region_group,
        "industry_group": industry_group,
    }

    result = predict(app.state.model, input_data)
    return {"success_probability": result}

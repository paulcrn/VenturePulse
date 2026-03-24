"""
SHAP-to-human-language translation module for VenturePulse survival predictions.

Converts raw XGBoost SHAP values (log-odds space, technical feature names) into
frontend-ready JSON with plain-English labels and impact levels.

Usage:
    from project_logic.shap_translator import explain_prediction
    result = explain_prediction(shap_values, feature_names, probability=0.87)
"""

# ---------------------------------------------------------------------------
# Feature label mappings
# ---------------------------------------------------------------------------

# Plain-English labels for every feature (19 numeric + 20 OHE after preprocessing).
# Used as default labels when direction-specific phrasing is not needed.
FEATURE_LABELS: dict[str, str] = {
    # --- Continuous funding metrics ---
    "funding_total_usd":        "Total capital raised",
    "funding_rounds":           "Number of funding rounds",
    "avg_raised_per_round":     "Average amount raised per round",

    # --- Binary funding type flags ---
    "seed":                     "Seed funding",
    "venture":                  "Venture capital backing",
    "angel":                    "Angel investment",
    "grant":                    "Grant funding",
    "debt_financing":           "Debt financing",
    "private_equity":           "Private equity backing",

    # --- Funding stage flags ---
    "round_A":                  "Series A funding",
    "round_B":                  "Series B funding",
    "round_C":                  "Series C funding",
    "round_D":                  "Series D funding",
    "round_E":                  "Series E+ funding",

    # --- Time / cadence metrics (direction-aware phrasing lives in DIRECTION_LABELS) ---
    "age_first_funding_days":   "Time to first funding",
    "has_multiple_rounds":      "Multiple funding rounds",
    "funding_span_days":        "Duration of funding activity",
    "avg_years_between_rounds": "Time between funding rounds",
    "time_since_last_funding":  "Time since last funding round",

    # --- Region OHE columns ---
    "region_group_USA":          "Based in USA",
    "region_group_EU_UK":        "Based in EU/UK",
    "region_group_Canada":       "Based in Canada",
    "region_group_Australia":    "Based in Australia",
    "region_group_Asia":         "Based in Asia",
    "region_group_Rest_World":   "Based in rest of world",
    "region_group_Rest_Americas": "Based in Latin America",
    "region_group_Unknown":      "Unknown region",       # excluded — see EXCLUDE_FEATURES

    # --- Industry OHE columns ---
    "industry_group_Health_Bio":       "Health & Biotech",
    "industry_group_Consumer_Internet": "Consumer Internet",
    "industry_group_Software_Data":    "Software & Data",
    "industry_group_Energy":           "Energy sector",
    "industry_group_Education":        "Education sector",
    "industry_group_Services":         "Services industry",
    "industry_group_Real_World":       "Real-world / physical industry",
    "industry_group_Ecommerce":        "E-commerce",
    "industry_group_FinTech":          "FinTech",
    "industry_group_Hardware_DeepTech": "Hardware & Deep Tech",
    "industry_group_Unknown":          "Unknown industry",  # excluded
    "industry_group_Other":            "Other industry",    # excluded
}

# ---------------------------------------------------------------------------
# Direction-aware labels
# For continuous features where the SHAP sign has a meaningful narrative
# interpretation, we override the default label based on whether SHAP is
# positive (pushes toward survival) or negative (pushes toward closure).
# ---------------------------------------------------------------------------

DIRECTION_LABELS: dict[str, dict[str, str]] = {
    # --- Continuous features ---
    "time_since_last_funding": {
        "positive": "Got funded recently",
        "negative": "Funding has gone stale",
    },
    "age_first_funding_days": {
        "positive": "Got funded quickly after founding",
        "negative": "Took a long time to get first funding",
    },
    "avg_years_between_rounds": {
        "positive": "Consistent pace between funding rounds",
        "negative": "Long gaps between funding rounds",
    },
    "funding_span_days": {
        "positive": "Long track record of sustained funding",
        "negative": "Short overall funding window",
    },

    # --- Binary funding type flags ---
    "seed": {
        "positive": "Has seed funding",
        "negative": "No seed funding on record",
    },
    "venture": {
        "positive": "Backed by venture capital",
        "negative": "No venture capital backing",
    },
    "angel": {
        "positive": "Has angel investment",
        "negative": "No angel investment",
    },
    "grant": {
        "positive": "Has received grant funding",
        "negative": "No grant funding",
    },
    "debt_financing": {
        "positive": "Has debt financing",
        "negative": "No debt financing",
    },
    "private_equity": {
        "positive": "Has private equity backing",
        "negative": "No private equity backing",
    },

    # --- Funding stage flags ---
    "round_A": {
        "positive": "Reached Series A",
        "negative": "Not reached Series A yet",
    },
    "round_B": {
        "positive": "Reached Series B",
        "negative": "Not reached Series B yet",
    },
    "round_C": {
        "positive": "Reached Series C",
        "negative": "Not reached Series C yet",
    },
    "round_D": {
        "positive": "Reached Series D",
        "negative": "Not reached Series D yet",
    },
}

# ---------------------------------------------------------------------------
# Features to exclude from the user-facing explanation
# (uninformative catch-all / unknown categories)
# ---------------------------------------------------------------------------

EXCLUDE_FEATURES: set[str] = {
    "region_group_Unknown",
    "industry_group_Unknown",
    "industry_group_Other",
}

# ---------------------------------------------------------------------------
# Impact thresholds (absolute SHAP magnitude, log-odds scale)
# ---------------------------------------------------------------------------
# Calibrated against mean |SHAP| from the test set:
#   time_since_last_funding ≈ 1.21  (top feature)
#   funding_total_usd       ≈ 0.49
#   age_first_funding_days  ≈ 0.23
#   funding_span_days       ≈ 0.15
#   round_A, round_B        ≈ 0.12
#   most others             < 0.08

_IMPACT_HIGH   = 0.40   # substantial shift in predicted probability
_IMPACT_MEDIUM = 0.10   # noticeable, but moderate


def _impact_level(abs_shap: float) -> str:
    if abs_shap >= _IMPACT_HIGH:
        return "high"
    if abs_shap >= _IMPACT_MEDIUM:
        return "medium"
    return "low"


def _resolve_label(feature: str, shap_value: float) -> str:
    """Return the best human-readable label for a feature given its SHAP direction."""
    if feature in DIRECTION_LABELS:
        direction = "positive" if shap_value >= 0 else "negative"
        return DIRECTION_LABELS[feature][direction]
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

# OHE feature prefixes — only the active category (value == 1) should be shown
_OHE_PREFIXES = ("region_group_", "industry_group_")


def explain_prediction(
    shap_values: "list[float] | Any",
    feature_names: "list[str]",
    probability: float,
    feature_values: "list[float] | Any | None" = None,
    top_n: int = 5,
) -> dict:
    """
    Translate raw SHAP output into a frontend-ready explanation dict.

    Parameters
    ----------
    shap_values : array-like of float
        Per-feature SHAP values in log-odds space (shape: [n_features]).
        Positive = pushes toward survival, negative = pushes toward closure.
    feature_names : list of str
        Feature names in the same order as shap_values.
        Must match the post-preprocessing feature names (e.g. 'region_group_EU_UK').
    probability : float
        Model's predicted survival probability in [0, 1].
    feature_values : array-like of float, optional
        Preprocessed feature values for this sample (same order as feature_names).
        Used to suppress inactive OHE categories (value == 0) so only the company's
        actual region/industry is surfaced, not confusing "not-X" signals.
    top_n : int
        Maximum number of strengths and risks to return (default 5 each).

    Returns
    -------
    dict with keys:
        prediction         – 'survived' or 'closed'
        confidence         – float, percentage confidence in that prediction (0–100)
        survival_probability – float, always the probability of survival (0–100)
        strengths          – list of {label, impact} dicts (positive SHAP, sorted by magnitude)
        risks              – list of {label, impact} dicts (negative SHAP, sorted by magnitude)
    """
    values_lookup = (
        dict(zip(feature_names, feature_values)) if feature_values is not None else {}
    )

    pairs = list(zip(feature_names, shap_values))

    # Filter out excluded features and inactive OHE categories
    def _keep(feat: str) -> bool:
        if feat in EXCLUDE_FEATURES:
            return False
        if any(feat.startswith(p) for p in _OHE_PREFIXES):
            return values_lookup.get(feat, 1) == 1  # only show the active category
        return True

    pairs = [(feat, val) for feat, val in pairs if _keep(feat)]

    # Sort by absolute SHAP descending
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    strengths = []
    risks = []

    for feat, val in pairs:
        if len(strengths) >= top_n and len(risks) >= top_n:
            break

        entry = {
            "label":  _resolve_label(feat, val),
            "impact": _impact_level(abs(val)),
        }

        if val > 0 and len(strengths) < top_n:
            strengths.append(entry)
        elif val < 0 and len(risks) < top_n:
            risks.append(entry)

    prediction = "survived" if probability >= 0.5 else "closed"
    confidence = round(
        (probability if prediction == "survived" else 1.0 - probability) * 100, 1
    )

    return {
        "prediction":          prediction,
        "confidence":          confidence,
        "survival_probability": round(probability * 100, 1),
        "strengths":           strengths,
        "risks":               risks,
    }


# ---------------------------------------------------------------------------
# Example calls (using representative SHAP values from the test set)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Feature names in post-preprocessing order (17 numeric + 20 OHE = 37 total)
    # round_E and has_multiple_rounds excluded after permutation importance analysis
    FEATURE_NAMES = [
        # numeric
        "funding_total_usd", "funding_rounds",
        "seed", "venture", "debt_financing", "angel", "grant", "private_equity",
        "round_A", "round_B", "round_C", "round_D",
        "avg_raised_per_round", "age_first_funding_days",
        "funding_span_days", "avg_years_between_rounds",
        "time_since_last_funding",
        # region OHE
        "region_group_USA", "region_group_EU_UK", "region_group_Canada",
        "region_group_Australia", "region_group_Asia", "region_group_Rest_World",
        "region_group_Rest_Americas", "region_group_Unknown",
        # industry OHE
        "industry_group_Health_Bio", "industry_group_Consumer_Internet",
        "industry_group_Software_Data", "industry_group_Unknown",
        "industry_group_Energy", "industry_group_Other", "industry_group_Education",
        "industry_group_Services", "industry_group_Real_World",
        "industry_group_Ecommerce", "industry_group_FinTech",
        "industry_group_Hardware_DeepTech",
    ]

    # ------------------------------------------------------------------
    # Example 1: High-confidence SURVIVED (Series B SaaS company, USA)
    # Characteristics: recent funding, raised a lot, multiple rounds
    # ------------------------------------------------------------------
    shap_survived = [
        0.41,   # funding_total_usd          (large total raise → good)
        0.09,   # funding_rounds
        0.00,   # seed
        0.02,   # venture
        0.00,   # debt_financing
        0.00,   # angel
        0.00,   # grant
        0.08,   # private_equity
        0.13,   # round_A
        0.12,   # round_B
        0.00,   # round_C
        0.00,   # round_D
        0.12,   # avg_raised_per_round
        0.38,   # age_first_funding_days     (funded quickly → positive)
        0.14,   # funding_span_days          (sustained → positive)
        0.03,   # avg_years_between_rounds
        1.82,   # time_since_last_funding    (very recent → large positive SHAP → strength)
        0.07,   # region_group_USA
        0.00,   # region_group_EU_UK
        0.00,   # region_group_Canada
        0.00,   # region_group_Australia
        0.00,   # region_group_Asia
        0.00,   # region_group_Rest_World
        0.00,   # region_group_Rest_Americas
        0.00,   # region_group_Unknown
        0.00,   # industry_group_Health_Bio
        0.00,   # industry_group_Consumer_Internet
        0.09,   # industry_group_Software_Data
        0.00,   # industry_group_Unknown
        0.00,   # industry_group_Energy
        0.00,   # industry_group_Other
        0.00,   # industry_group_Education
        0.00,   # industry_group_Services
        0.00,   # industry_group_Real_World
        0.00,   # industry_group_Ecommerce
        0.00,   # industry_group_FinTech
        0.00,   # industry_group_Hardware_DeepTech
    ]
    print("=" * 60)
    print("Example 1: High-confidence SURVIVED (SaaS/USA, Series B)")
    print("=" * 60)
    result1 = explain_prediction(shap_survived, FEATURE_NAMES, probability=0.987)
    print(json.dumps(result1, indent=2))

    # ------------------------------------------------------------------
    # Example 2: High-confidence CLOSED (no recent funding, early stage)
    # Characteristics: stale funding, no series A, small total raise
    # ------------------------------------------------------------------
    shap_closed = [
       -0.31,   # funding_total_usd          (low total raise → negative)
       -0.07,   # funding_rounds
        0.00,   # seed
       -0.02,   # venture
        0.00,   # debt_financing
        0.00,   # angel
        0.00,   # grant
        0.00,   # private_equity
       -0.11,   # round_A                    (no Series A → negative)
        0.00,   # round_B
        0.00,   # round_C
        0.00,   # round_D
       -0.04,   # avg_raised_per_round
       -0.15,   # age_first_funding_days     (slow to first funding → negative)
       -0.12,   # funding_span_days          (short span → negative)
       -0.02,   # avg_years_between_rounds
       -1.74,   # time_since_last_funding    (very stale → large negative SHAP → risk)
        0.00,   # region_group_USA
        0.05,   # region_group_EU_UK
        0.00,   # region_group_Canada
        0.00,   # region_group_Australia
        0.00,   # region_group_Asia
        0.00,   # region_group_Rest_World
        0.00,   # region_group_Rest_Americas
        0.00,   # region_group_Unknown
        0.00,   # industry_group_Health_Bio
        0.00,   # industry_group_Consumer_Internet
        0.00,   # industry_group_Software_Data
        0.00,   # industry_group_Unknown
        0.00,   # industry_group_Energy
        0.00,   # industry_group_Other
        0.00,   # industry_group_Education
       -0.06,   # industry_group_Services
        0.00,   # industry_group_Real_World
        0.00,   # industry_group_Ecommerce
        0.00,   # industry_group_FinTech
        0.00,   # industry_group_Hardware_DeepTech
    ]

    print()
    print("=" * 60)
    print("Example 2: High-confidence CLOSED (stale funding, EU/UK)")
    print("=" * 60)
    result2 = explain_prediction(shap_closed, FEATURE_NAMES, probability=0.083)
    print(json.dumps(result2, indent=2))

    # ------------------------------------------------------------------
    # Example 3: Borderline case (near 50% probability)
    # ------------------------------------------------------------------
    shap_borderline = [
        0.09,   # funding_total_usd
        0.04,   # funding_rounds
        0.00,   # seed
        0.01,   # venture
        0.00,   # debt_financing
        0.00,   # angel
        0.00,   # grant
        0.00,   # private_equity
        0.10,   # round_A
        0.00,   # round_B
        0.00,   # round_C
        0.00,   # round_D
        0.03,   # avg_raised_per_round
        0.11,   # age_first_funding_days
       -0.08,   # funding_span_days
        0.02,   # avg_years_between_rounds
        0.14,   # time_since_last_funding   (moderately recent → positive SHAP → mild strength)
        0.00,   # region_group_USA
        0.00,   # region_group_EU_UK
        0.03,   # region_group_Canada
        0.00,   # region_group_Australia
        0.00,   # region_group_Asia
        0.00,   # region_group_Rest_World
        0.00,   # region_group_Rest_Americas
        0.00,   # region_group_Unknown
        0.00,   # industry_group_Health_Bio
        0.00,   # industry_group_Consumer_Internet
        0.00,   # industry_group_Software_Data
        0.00,   # industry_group_Unknown
        0.00,   # industry_group_Energy
        0.00,   # industry_group_Other
        0.00,   # industry_group_Education
        0.00,   # industry_group_Services
        0.00,   # industry_group_Real_World
       -0.08,   # industry_group_Ecommerce
        0.00,   # industry_group_FinTech
        0.00,   # industry_group_Hardware_DeepTech
    ]

    print()
    print("=" * 60)
    print("Example 3: Borderline case (~52% survival probability)")
    print("=" * 60)
    result3 = explain_prediction(shap_borderline, FEATURE_NAMES, probability=0.523)
    print(json.dumps(result3, indent=2))

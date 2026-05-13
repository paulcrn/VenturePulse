"""Unit tests for the SHAP-to-plain-English translation layer.

These tests cover the pure-Python translation logic without requiring the
trained pipeline artifact, so CI can run them in a fresh environment.
"""

from project_logic.shap_translator import explain_prediction


FEATURE_NAMES = [
    "funding_total_usd", "funding_rounds",
    "seed", "venture", "debt_financing", "angel", "grant", "private_equity",
    "round_A", "round_B", "round_C", "round_D",
    "avg_raised_per_round", "age_first_funding_days",
    "funding_span_days", "avg_years_between_rounds",
    "time_since_last_funding",
    "region_group_USA", "region_group_EU_UK", "region_group_Canada",
    "region_group_Australia", "region_group_Asia", "region_group_Rest_World",
    "region_group_Rest_Americas", "region_group_Unknown",
    "industry_group_Health_Bio", "industry_group_Consumer_Internet",
    "industry_group_Software_Data", "industry_group_Unknown",
    "industry_group_Energy", "industry_group_Other", "industry_group_Education",
    "industry_group_Services", "industry_group_Real_World",
    "industry_group_Ecommerce", "industry_group_FinTech",
    "industry_group_Hardware_DeepTech",
]


def _zero_shap():
    return [0.0] * len(FEATURE_NAMES)


def test_survived_branch_emits_strengths():
    shap_values = _zero_shap()
    shap_values[0] = 0.5  # funding_total_usd → positive → strength

    result = explain_prediction(shap_values, FEATURE_NAMES, probability=0.9)

    assert result["prediction"] == "survived"
    assert result["confidence"] == 90.0
    assert result["survival_probability"] == 90.0
    assert any(s["label"] == "Total capital raised" for s in result["strengths"])
    assert result["risks"] == []


def test_closed_branch_emits_risks_and_confidence_is_for_predicted_class():
    shap_values = _zero_shap()
    shap_values[0] = -0.5  # funding_total_usd → negative → risk

    result = explain_prediction(shap_values, FEATURE_NAMES, probability=0.1)

    assert result["prediction"] == "closed"
    # Confidence reports certainty in the *predicted* class
    assert result["confidence"] == 90.0
    # survival_probability is always the survival probability
    assert result["survival_probability"] == 10.0
    assert len(result["risks"]) >= 1
    assert result["strengths"] == []


def test_high_magnitude_shap_yields_high_impact():
    shap_values = _zero_shap()
    # time_since_last_funding at index 16; 1.5 is well above the high threshold (0.40)
    shap_values[16] = 1.5

    result = explain_prediction(shap_values, FEATURE_NAMES, probability=0.95)

    top = result["strengths"][0]
    assert top["impact"] == "high"
    # Direction-aware label: positive SHAP on this feature → "Active funding momentum"
    assert top["label"] == "Active funding momentum"


def test_direction_aware_label_flips_with_shap_sign():
    shap_values = _zero_shap()
    # Negative SHAP on time_since_last_funding → "No recent funding activity"
    shap_values[16] = -1.5

    result = explain_prediction(shap_values, FEATURE_NAMES, probability=0.05)

    top_risk = result["risks"][0]
    assert top_risk["label"] == "No recent funding activity"


def test_ohe_only_active_category_is_surfaced():
    # Both USA and EU_UK have a positive SHAP, but only USA is the active region.
    # The translator must suppress EU_UK (inactive OHE category).
    shap_values = _zero_shap()
    shap_values[17] = 0.3  # region_group_USA
    shap_values[18] = 0.3  # region_group_EU_UK

    feature_values = [0.0] * len(FEATURE_NAMES)
    feature_values[17] = 1.0  # USA active
    feature_values[18] = 0.0  # EU/UK inactive

    result = explain_prediction(
        shap_values,
        FEATURE_NAMES,
        probability=0.8,
        feature_values=feature_values,
    )

    labels = [s["label"] for s in result["strengths"]]
    assert "Based in USA" in labels
    assert "Based in EU/UK" not in labels


def test_binary_feature_absent_with_positive_shap_is_suppressed():
    # round_A=0 (no Series A) + positive SHAP would label "Reached Series A",
    # which contradicts the input. The translator must drop it.
    shap_values = _zero_shap()
    shap_values[8] = 0.3  # round_A SHAP positive

    raw_input = {"round_A": 0}

    result = explain_prediction(
        shap_values,
        FEATURE_NAMES,
        probability=0.7,
        raw_input=raw_input,
    )

    labels = [s["label"] for s in result["strengths"]]
    assert "Reached Series A" not in labels


def test_excluded_features_never_appear():
    # Unknown region/industry/Other are explicitly excluded from the explanation.
    shap_values = _zero_shap()
    shap_values[24] = 1.0  # region_group_Unknown (excluded)
    shap_values[28] = 1.0  # industry_group_Unknown (excluded)

    feature_values = [0.0] * len(FEATURE_NAMES)
    feature_values[24] = 1.0
    feature_values[28] = 1.0

    result = explain_prediction(
        shap_values,
        FEATURE_NAMES,
        probability=0.9,
        feature_values=feature_values,
    )

    labels = [s["label"] for s in result["strengths"]]
    assert "Unknown region" not in labels
    assert "Unknown industry" not in labels

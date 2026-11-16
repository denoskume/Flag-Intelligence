"""Tests for duplicate ISO labels sharing the same French national flag."""
from flag_recognition.shared_flags import resolve_french_flag_design


def test_user_reported_mayotte_top1_is_french_tricolor_design():
    # Exact percentages displayed in the user's failed recognition screenshot.
    # They are model outputs; their sum is not a calibrated probability.
    prediction = (
        ("yt", 0.540),
        ("mf", 0.125),
        ("bl", 0.123),
        ("re", 0.120),
        ("fr", 0.063),
    )
    decision = resolve_french_flag_design(prediction)
    assert decision is not None
    assert decision.code == "fr"
    assert decision.design_name == "tricolore français"
    assert abs(decision.combined_score - 0.971) < 1e-9


def test_single_mayotte_prediction_does_not_turn_into_france():
    assert resolve_french_flag_design([
        ("yt", 0.96), ("fr", 0.01), ("mf", 0.01)
    ]) is None


def test_unrelated_flag_predictions_never_canonicalized():
    assert resolve_french_flag_design([
        ("ie", 0.62), ("ci", 0.20), ("fr", 0.09),
        ("mf", 0.04), ("yt", 0.02)
    ]) is None


def test_mixed_territory_predictions_do_not_claim_french_flag():
    assert resolve_french_flag_design([
        ("yt", 0.54), ("mf", 0.18), ("fr", 0.09),
        ("ie", 0.11), ("bl", 0.04)
    ]) is None


def test_duplicate_labels_are_handled_case_insensitively():
    result = resolve_french_flag_design([
        ("YT", 0.50), ("MF", 0.21), ("RE", 0.19), ("FR", 0.04)
    ])
    assert result is not None
    assert result.code == "fr"

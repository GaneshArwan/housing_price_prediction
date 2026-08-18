import pytest
from src.guardrails import PredictionGuardrail


def test_valid_prediction():
    assert PredictionGuardrail.validate_prediction(200000.0) is True


def test_zero_price_fails():
    assert PredictionGuardrail.validate_prediction(0) is False


def test_negative_price_fails():
    assert PredictionGuardrail.validate_prediction(-50000) is False


def test_extreme_high_price_fails():
    """Prices above $10M should fail the business guardrail."""
    assert PredictionGuardrail.validate_prediction(15000000) is False


def test_extreme_low_price_fails():
    """Prices below $1,000 should fail the business guardrail."""
    assert PredictionGuardrail.validate_prediction(500) is False


def test_boundary_low_passes():
    assert PredictionGuardrail.validate_prediction(1000) is True


def test_boundary_high_passes():
    assert PredictionGuardrail.validate_prediction(10000000) is True

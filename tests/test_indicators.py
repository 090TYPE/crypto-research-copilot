"""Pure indicator math — no I/O, no mocks."""

import pytest

from app.agent.indicators import rsi, sma


def test_sma_basic() -> None:
    assert sma([1, 2, 3, 4, 5], 5) == 3.0
    assert sma([10, 20, 30], 2) == 25.0


def test_sma_too_few() -> None:
    with pytest.raises(ValueError):
        sma([1, 2], 5)


def test_rsi_all_gains_is_100() -> None:
    prices = [float(i) for i in range(1, 20)]  # strictly increasing
    assert rsi(prices, 14) == 100.0


def test_rsi_bounds_and_value() -> None:
    prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
              46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    value = rsi(prices, 14)
    assert 0.0 <= value <= 100.0
    assert value > 50.0  # net upward series


def test_rsi_too_few() -> None:
    with pytest.raises(ValueError):
        rsi([1, 2, 3], 14)

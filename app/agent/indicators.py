"""Pure technical-indicator math. No I/O — trivially unit-testable."""


def sma(values: list[float], period: int) -> float:
    """Simple moving average of the last `period` values."""
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        raise ValueError(f"need >= {period} values, got {len(values)}")
    window = values[-period:]
    return sum(window) / period


def rsi(values: list[float], period: int = 14) -> float:
    """Relative Strength Index (Wilder's simple-average variant), 0..100.

    Needs at least `period + 1` values to form `period` deltas.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period + 1:
        raise ValueError(f"need >= {period + 1} values, got {len(values)}")

    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    window = deltas[-period:]
    gains = [d for d in window if d > 0]
    losses = [-d for d in window if d < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

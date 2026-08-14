import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def _prepare_series(data):
    """
    Convert input data to a clean numeric pandas Series.

    Parameters
    ----------
    data : array-like or pandas Series
        Historical demand observations.

    Returns
    -------
    pandas.Series
        Clean numeric series with missing values removed.
    """
    series = pd.Series(data, dtype="float64").dropna().reset_index(drop=True)

    if len(series) == 0:
        raise ValueError("The historical demand series is empty.")

    return series


def moving_average_forecast(data, window, horizon=6):
    """
    Generate a recursive simple moving-average forecast.

    For multi-period forecasting, each newly forecast value is appended
    to the history and can therefore be used to forecast subsequent periods.

    Parameters
    ----------
    data : array-like or pandas Series
        Historical demand data.
    window : int
        Number of most recent periods used in the moving average.
    horizon : int, default=6
        Number of future periods to forecast.

    Returns
    -------
    numpy.ndarray
        Forecasts for the next `horizon` periods.
    """
    series = _prepare_series(data)

    if window < 1:
        raise ValueError("Moving-average window must be at least 1.")

    if window > len(series):
        raise ValueError(
            "Moving-average window cannot exceed the number "
            "of historical observations."
        )

    history = series.tolist()
    forecasts = []

    for _ in range(horizon):
        forecast = np.mean(history[-window:])
        forecasts.append(forecast)
        history.append(forecast)

    return np.array(forecasts)


def weighted_moving_average_forecast(data, weights, horizon=6):
    """
    Generate a recursive weighted moving-average forecast.

    The first weight corresponds to the MOST RECENT observation.

    Example
    -------
    weights = [0.5, 0.3, 0.2]

    means:
        50% weight on the most recent month,
        30% on two months ago,
        20% on three months ago.

    Parameters
    ----------
    data : array-like or pandas Series
        Historical demand data.
    weights : array-like
        Weights from most recent to oldest observation.
        Weights must sum to 1.
    horizon : int, default=6
        Number of future periods to forecast.

    Returns
    -------
    numpy.ndarray
        Forecasts for the next `horizon` periods.
    """
    series = _prepare_series(data)
    weights = np.asarray(weights, dtype=float)

    if len(weights) == 0:
        raise ValueError("At least one weight must be provided.")

    if len(weights) > len(series):
        raise ValueError(
            "The number of weights cannot exceed the number "
            "of historical observations."
        )

    if np.any(weights < 0):
        raise ValueError("Weights cannot be negative.")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must sum to 1.")

    history = series.tolist()
    forecasts = []

    for _ in range(horizon):
        recent_values = np.array(
            history[-len(weights):][::-1]
        )

        forecast = np.sum(recent_values * weights)

        forecasts.append(forecast)
        history.append(forecast)

    return np.array(forecasts)


def exponential_smoothing_forecast(data, alpha, horizon=6):
    """
    Generate forecasts using simple exponential smoothing.

    Students explicitly select alpha rather than allowing the software
    to optimize it.

    Parameters
    ----------
    data : array-like or pandas Series
        Historical demand data.
    alpha : float
        Smoothing constant between 0 and 1.
    horizon : int, default=6
        Number of future periods to forecast.

    Returns
    -------
    numpy.ndarray
        Forecasts for the next `horizon` periods.
    """
    series = _prepare_series(data)

    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be greater than 0 and no greater than 1.")

    # Initial level
    level = series.iloc[0]

    # Update level recursively through historical observations
    for actual in series.iloc[1:]:
        level = alpha * actual + (1 - alpha) * level

    # Simple exponential smoothing produces the same forecast
    # for every future period when no trend or seasonality is included.
    forecasts = np.repeat(level, horizon)

    return forecasts


def arima_forecast(data, order, horizon=6):
    """
    Generate forecasts using a constrained ARIMA model.

    Parameters
    ----------
    data : array-like or pandas Series
        Historical demand data.
    order : tuple
        ARIMA order (p, d, q), for example (1, 1, 0).
    horizon : int, default=6
        Number of future periods to forecast.

    Returns
    -------
    numpy.ndarray
        Forecasts for the next `horizon` periods.
    """
    series = _prepare_series(data)

    if (
        not isinstance(order, tuple)
        or len(order) != 3
        or not all(isinstance(x, int) for x in order)
    ):
        raise ValueError("ARIMA order must be a tuple of three integers: (p, d, q).")

    p, d, q = order

    if p < 0 or d < 0 or q < 0:
        raise ValueError("ARIMA parameters cannot be negative.")

    model = ARIMA(series, order=order)
    fitted_model = model.fit()

    forecasts = fitted_model.forecast(steps=horizon)

    return np.asarray(forecasts, dtype=float)


# ARIMA models that students are allowed to choose
ARIMA_CHOICES = {
    "ARIMA(1,0,0)": (1, 0, 0),
    "ARIMA(1,1,0)": (1, 1, 0),
    "ARIMA(0,1,1)": (0, 1, 1),
    "ARIMA(1,1,1)": (1, 1, 1),
}
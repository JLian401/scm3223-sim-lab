import numpy as np
import pandas as pd


def _prepare_actual_forecast(actual, forecast):
    """
    Convert actual and forecast values to clean numpy arrays
    and verify that they have the same length.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError(
            "Actual and forecast arrays must have the same length."
        )

    if len(actual) == 0:
        raise ValueError(
            "Actual and forecast arrays cannot be empty."
        )

    if np.isnan(actual).any() or np.isnan(forecast).any():
        raise ValueError(
            "Actual and forecast values cannot contain missing values."
        )

    return actual, forecast


def forecast_errors(actual, forecast):
    """
    Calculate signed forecast errors.

    Error is defined as:

        Actual - Forecast

    Therefore:
        positive error = underforecast
        negative error = overforecast
    """
    actual, forecast = _prepare_actual_forecast(actual, forecast)

    return actual - forecast


def mean_forecast_error(actual, forecast):
    """
    Mean Forecast Error (MFE).

    Positive MFE indicates overall underforecasting.
    Negative MFE indicates overall overforecasting.
    """
    errors = forecast_errors(actual, forecast)

    return np.mean(errors)


def mean_squared_error(actual, forecast):
    """
    Mean Squared Error (MSE).
    """
    errors = forecast_errors(actual, forecast)

    return np.mean(errors ** 2)


def mean_absolute_error(actual, forecast):
    """
    Mean Absolute Error (MAE), also commonly called
    Mean Absolute Deviation (MAD) in forecasting applications.
    """
    errors = forecast_errors(actual, forecast)

    return np.mean(np.abs(errors))


def mean_absolute_percentage_error(actual, forecast):
    """
    Mean Absolute Percentage Error (MAPE).

    Observations with actual demand equal to zero are excluded
    because percentage error is undefined when actual = 0.

    Returns
    -------
    float
        MAPE expressed as a percentage.
    """
    actual, forecast = _prepare_actual_forecast(actual, forecast)

    nonzero = actual != 0

    if not np.any(nonzero):
        return np.nan

    percentage_errors = (
        np.abs(actual[nonzero] - forecast[nonzero])
        / np.abs(actual[nonzero])
    )

    return np.mean(percentage_errors) * 100


def cumulative_forecast_error(actual, forecast):
    """
    Cumulative Forecast Error (CFE).

    Positive CFE indicates cumulative underforecasting.
    Negative CFE indicates cumulative overforecasting.
    """
    errors = forecast_errors(actual, forecast)

    return np.sum(errors)


def tracking_signal(actual, forecast):
    """
    Tracking Signal.

    Defined as:

        Cumulative Forecast Error / MAD

    Positive tracking signal indicates a tendency to underforecast.
    Negative tracking signal indicates a tendency to overforecast.
    """
    errors = forecast_errors(actual, forecast)

    mad = np.mean(np.abs(errors))

    if np.isclose(mad, 0):
        return 0.0

    cfe = np.sum(errors)

    return cfe / mad


def calculate_metrics(actual, forecast):
    """
    Calculate all forecasting performance measures.

    Returns
    -------
    dict
        Dictionary containing:
        MFE, MSE, MAE/MAD, MAPE, CFE, and Tracking Signal.
    """
    return {
        "MFE": mean_forecast_error(actual, forecast),
        "MSE": mean_squared_error(actual, forecast),
        "MAE": mean_absolute_error(actual, forecast),
        "MAPE": mean_absolute_percentage_error(actual, forecast),
        "CFE": cumulative_forecast_error(actual, forecast),
        "Tracking Signal": tracking_signal(actual, forecast),
    }


def create_error_table(actual, forecast, periods=None):
    """
    Create a period-by-period forecast evaluation table.

    Parameters
    ----------
    actual : array-like
        Actual demand.
    forecast : array-like
        Forecast demand.
    periods : array-like, optional
        Month labels or period labels.

    Returns
    -------
    pandas.DataFrame
        Detailed error table.
    """
    actual, forecast = _prepare_actual_forecast(actual, forecast)

    errors = actual - forecast
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

    percentage_errors = np.where(
        actual != 0,
        absolute_errors / np.abs(actual) * 100,
        np.nan,
    )

    if periods is None:
        periods = np.arange(1, len(actual) + 1)

    table = pd.DataFrame(
        {
            "Period": periods,
            "Actual": actual,
            "Forecast": forecast,
            "Error": errors,
            "Absolute Error": absolute_errors,
            "Squared Error": squared_errors,
            "Absolute Percentage Error": percentage_errors,
        }
    )

    return table
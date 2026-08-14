import numpy as np
import pandas as pd

from simulations.forecasting.data import (
    prepare_simulation_data,
    get_series,
)

from simulations.forecasting.methods import (
    moving_average_forecast,
    weighted_moving_average_forecast,
    exponential_smoothing_forecast,
    arima_forecast,
    ARIMA_CHOICES,
)

from simulations.forecasting.metrics import (
    calculate_metrics,
    create_error_table,
)


# ---------------------------------------------------------
# Student-facing method names
# ---------------------------------------------------------

METHOD_MOVING_AVERAGE = "Moving Average"
METHOD_WEIGHTED_MOVING_AVERAGE = "Weighted Moving Average"
METHOD_EXPONENTIAL_SMOOTHING = "Exponential Smoothing"
METHOD_ARIMA = "ARIMA"

VALID_METHODS = [
    METHOD_MOVING_AVERAGE,
    METHOD_WEIGHTED_MOVING_AVERAGE,
    METHOD_EXPONENTIAL_SMOOTHING,
    METHOD_ARIMA,
]


# ---------------------------------------------------------
# General utilities
# ---------------------------------------------------------

def _validate_region(region):
    """
    Validate forecasting region.
    """

    valid_regions = [
        "Aggregate",
        "NY",
        "CO",
        "TX",
    ]

    if region not in valid_regions:
        raise ValueError(
            f"region must be one of {valid_regions}"
        )


def _validate_horizon(horizon):
    """
    Validate forecast horizon.
    """

    if not isinstance(horizon, int):
        raise ValueError(
            "Forecast horizon must be an integer."
        )

    if horizon < 1:
        raise ValueError(
            "Forecast horizon must be at least 1."
        )


# ---------------------------------------------------------
# Unified forecasting function
# ---------------------------------------------------------

def generate_forecast(
    historical_data,
    method,
    horizon=6,
    **parameters,
):
    """
    Generate forecasts using any student-facing forecasting method.

    Parameters
    ----------
    historical_data : array-like or pandas Series
        Historical demand proxy observations.

    method : str
        One of:
            Moving Average
            Weighted Moving Average
            Exponential Smoothing
            ARIMA

    horizon : int, default=6
        Number of periods to forecast.

    **parameters
        Parameters depend on forecasting method.

        Moving Average:
            window=int

        Weighted Moving Average:
            weights=list

        Exponential Smoothing:
            alpha=float

        ARIMA:
            order=tuple
            OR
            arima_name=str

    Returns
    -------
    numpy.ndarray
        Forecast values.
    """

    _validate_horizon(horizon)

    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown forecasting method: {method}"
        )

    # -----------------------------------------------------
    # Moving Average
    # -----------------------------------------------------

    if method == METHOD_MOVING_AVERAGE:

        if "window" not in parameters:
            raise ValueError(
                "Moving Average requires 'window'."
            )

        return moving_average_forecast(
            historical_data,
            window=int(parameters["window"]),
            horizon=horizon,
        )

    # -----------------------------------------------------
    # Weighted Moving Average
    # -----------------------------------------------------

    if method == METHOD_WEIGHTED_MOVING_AVERAGE:

        if "weights" not in parameters:
            raise ValueError(
                "Weighted Moving Average requires 'weights'."
            )

        return weighted_moving_average_forecast(
            historical_data,
            weights=parameters["weights"],
            horizon=horizon,
        )

    # -----------------------------------------------------
    # Exponential Smoothing
    # -----------------------------------------------------

    if method == METHOD_EXPONENTIAL_SMOOTHING:

        if "alpha" not in parameters:
            raise ValueError(
                "Exponential Smoothing requires 'alpha'."
            )

        return exponential_smoothing_forecast(
            historical_data,
            alpha=float(parameters["alpha"]),
            horizon=horizon,
        )

    # -----------------------------------------------------
    # ARIMA
    # -----------------------------------------------------

    if method == METHOD_ARIMA:

        # Allow either:
        # order=(0,1,1)
        #
        # or:
        # arima_name="ARIMA(0,1,1)"

        if "order" in parameters:

            order = parameters["order"]

        elif "arima_name" in parameters:

            arima_name = parameters["arima_name"]

            if arima_name not in ARIMA_CHOICES:
                raise ValueError(
                    f"Unknown ARIMA choice: {arima_name}"
                )

            order = ARIMA_CHOICES[arima_name]

        else:
            raise ValueError(
                "ARIMA requires either 'order' "
                "or 'arima_name'."
            )

        return arima_forecast(
            historical_data,
            order=order,
            horizon=horizon,
        )


# ---------------------------------------------------------
# Validation stage
# ---------------------------------------------------------

def run_validation(
    method,
    region="Aggregate",
    training_window="all",
    **parameters,
):
    """
    Run one forecasting model against the fixed
    Jul-Dec 2024 validation period.

    Students may experiment repeatedly during this stage.

    Parameters
    ----------
    method : str
        Forecasting method.

    region : str, default="Aggregate"
        Aggregate, NY, CO, or TX.

    training_window : int or "all"
        Amount of training data supplied to the method.

        In the final student interface, this will mainly be
        relevant when explicitly desired. It remains available
        here for flexibility and diagnostics.

    **parameters
        Forecasting method parameters.

    Returns
    -------
    dict
        Forecasts, actuals, metrics, error table,
        and metadata.
    """

    _validate_region(region)

    sim_data = prepare_simulation_data(
        training_window=training_window
    )

    train_df = sim_data["train_selected"]
    validation_df = sim_data["validation"]

    train_series = get_series(
        train_df,
        region=region,
    )

    actual = get_series(
        validation_df,
        region=region,
    ).to_numpy()

    forecast = generate_forecast(
        historical_data=train_series,
        method=method,
        horizon=len(actual),
        **parameters,
    )

    metrics = calculate_metrics(
        actual,
        forecast,
    )

    error_table = create_error_table(
        actual,
        forecast,
        periods=validation_df[
            "Year-Month"
        ].dt.strftime("%b %Y"),
    )

    return {
        "stage": "validation",
        "region": region,
        "method": method,
        "parameters": parameters,
        "training_window": training_window,
        "training_start": train_df[
            "Year-Month"
        ].min(),
        "training_end": train_df[
            "Year-Month"
        ].max(),
        "forecast_dates": validation_df[
            "Year-Month"
        ].copy(),
        "forecast": np.asarray(
            forecast,
            dtype=float,
        ),
        "actual": np.asarray(
            actual,
            dtype=float,
        ),
        "metrics": metrics,
        "error_table": error_table,
    }


# ---------------------------------------------------------
# Final-test forecast
# ---------------------------------------------------------

def run_final_forecast(
    method,
    region="Aggregate",
    **parameters,
):
    """
    Generate the locked final forecast for Jan-Jun 2025.

    IMPORTANT:
    The model is refit using all data available through
    Dec 2024:

        Apr 2020 - Jun 2024 training
        +
        Jul 2024 - Dec 2024 validation

    The Jan-Jun 2025 actual observations are NOT used in
    generating the forecast.

    This function returns forecasts and forecast dates,
    but intentionally does not calculate test metrics.

    Test metrics should only be calculated after the
    student chooses to reveal the final actual demand.
    """

    _validate_region(region)

    sim_data = prepare_simulation_data(
        training_window="all"
    )

    train_df = sim_data["train_full"]
    validation_df = sim_data["validation"]
    test_df = sim_data["test"]

    # Information that would genuinely be available
    # at the end of Dec 2024
    final_history_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    historical_series = get_series(
        final_history_df,
        region=region,
    )

    horizon = len(test_df)

    forecast = generate_forecast(
        historical_data=historical_series,
        method=method,
        horizon=horizon,
        **parameters,
    )

    return {
        "stage": "final_forecast",
        "region": region,
        "method": method,
        "parameters": parameters,
        "history_start": final_history_df[
            "Year-Month"
        ].min(),
        "history_end": final_history_df[
            "Year-Month"
        ].max(),
        "forecast_dates": test_df[
            "Year-Month"
        ].copy(),
        "forecast": np.asarray(
            forecast,
            dtype=float,
        ),
    }


# ---------------------------------------------------------
# Reveal/evaluate final test period
# ---------------------------------------------------------

def evaluate_final_forecast(
    final_forecast_result,
):
    """
    Reveal Jan-Jun 2025 actual registrations and calculate
    final out-of-sample forecast accuracy.

    Parameters
    ----------
    final_forecast_result : dict
        Output from run_final_forecast().

    Returns
    -------
    dict
        Original final forecast information plus actual
        values, performance metrics, and detailed errors.
    """

    if (
        final_forecast_result.get("stage")
        != "final_forecast"
    ):
        raise ValueError(
            "Input must be the result of "
            "run_final_forecast()."
        )

    region = final_forecast_result["region"]

    sim_data = prepare_simulation_data(
        training_window="all"
    )

    test_df = sim_data["test"]

    actual = get_series(
        test_df,
        region=region,
    ).to_numpy()

    forecast = np.asarray(
        final_forecast_result["forecast"],
        dtype=float,
    )

    metrics = calculate_metrics(
        actual,
        forecast,
    )

    error_table = create_error_table(
        actual,
        forecast,
        periods=test_df[
            "Year-Month"
        ].dt.strftime("%b %Y"),
    )

    result = final_forecast_result.copy()

    result.update(
        {
            "stage": "final_revealed",
            "actual": np.asarray(
                actual,
                dtype=float,
            ),
            "metrics": metrics,
            "error_table": error_table,
        }
    )

    return result


# ---------------------------------------------------------
# Judgmental forecast evaluation
# ---------------------------------------------------------

def evaluate_judgmental_forecast(
    forecasts,
    stage="validation",
    region="Aggregate",
):
    """
    Evaluate a student's manually entered six-month forecast.

    This lets us compare judgmental forecasting with
    statistical methods.

    Parameters
    ----------
    forecasts : array-like
        Student-entered forecasts.

    stage : str
        "validation" or "test"

    region : str
        Aggregate, NY, CO, or TX.

    Returns
    -------
    dict
    """

    _validate_region(region)

    forecasts = np.asarray(
        forecasts,
        dtype=float,
    )

    sim_data = prepare_simulation_data(
        training_window="all"
    )

    if stage == "validation":
        actual_df = sim_data["validation"]

    elif stage == "test":
        actual_df = sim_data["test"]

    else:
        raise ValueError(
            "stage must be 'validation' or 'test'."
        )

    actual = get_series(
        actual_df,
        region=region,
    ).to_numpy()

    if len(forecasts) != len(actual):
        raise ValueError(
            f"Expected {len(actual)} judgmental forecasts, "
            f"but received {len(forecasts)}."
        )

    if np.isnan(forecasts).any():
        raise ValueError(
            "Judgmental forecasts cannot contain missing values."
        )

    if np.any(forecasts < 0):
        raise ValueError(
            "Judgmental forecasts cannot be negative."
        )

    metrics = calculate_metrics(
        actual,
        forecasts,
    )

    error_table = create_error_table(
        actual,
        forecasts,
        periods=actual_df[
            "Year-Month"
        ].dt.strftime("%b %Y"),
    )

    return {
        "stage": stage,
        "region": region,
        "forecast_dates": actual_df[
            "Year-Month"
        ].copy(),
        "forecast": forecasts,
        "actual": actual,
        "metrics": metrics,
        "error_table": error_table,
    }
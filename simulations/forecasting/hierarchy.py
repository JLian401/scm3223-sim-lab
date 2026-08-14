import numpy as np
import pandas as pd

from simulations.forecasting.metrics import calculate_metrics


VALID_STATES = ["NY", "CO", "TX"]


# =========================================================
# Utilities
# =========================================================

def _validate_states(df):
    """
    Check that NY, CO, and TX are present.
    """

    missing = [
        state
        for state in VALID_STATES
        if state not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required state columns: "
            + ", ".join(missing)
        )


def _validate_aggregate_forecast(aggregate_forecast):
    """
    Convert aggregate forecasts to a numeric numpy array.
    """

    forecast = np.asarray(
        aggregate_forecast,
        dtype=float,
    )

    if forecast.ndim != 1:
        raise ValueError(
            "Aggregate forecast must be one-dimensional."
        )

    if len(forecast) == 0:
        raise ValueError(
            "Aggregate forecast cannot be empty."
        )

    if np.isnan(forecast).any():
        raise ValueError(
            "Aggregate forecast cannot contain missing values."
        )

    if np.any(forecast < 0):
        raise ValueError(
            "Aggregate forecast cannot contain negative values."
        )

    return forecast


# =========================================================
# State-share calculations
# =========================================================

def historical_state_shares(
    historical_df,
):
    """
    Calculate state shares using all observations in the
    supplied historical period.

    State share is based on cumulative registrations:

        state registrations
        -------------------
        NY + CO + TX registrations

    Parameters
    ----------
    historical_df : pandas.DataFrame
        Historical data containing NY, CO, and TX.

    Returns
    -------
    dict
        Example:
        {
            "NY": 0.25,
            "CO": 0.20,
            "TX": 0.55
        }
    """

    _validate_states(
        historical_df
    )

    state_totals = (
        historical_df[
            VALID_STATES
        ]
        .sum()
    )

    total = state_totals.sum()

    if total <= 0:
        raise ValueError(
            "Historical state registrations must sum "
            "to a positive value."
        )

    shares = (
        state_totals
        / total
    )

    return shares.to_dict()


def recent_state_shares(
    historical_df,
    months=3,
):
    """
    Calculate state shares based on the most recent
    observations.

    Default rule:
        most recent 3 months

    Parameters
    ----------
    historical_df : pandas.DataFrame

    months : int, default=3
        Number of recent months used to calculate shares.

    Returns
    -------
    dict
    """

    _validate_states(
        historical_df
    )

    if not isinstance(
        months,
        int,
    ):
        raise ValueError(
            "months must be an integer."
        )

    if months < 1:
        raise ValueError(
            "months must be at least 1."
        )

    if months > len(
        historical_df
    ):
        raise ValueError(
            f"Requested {months} months, but only "
            f"{len(historical_df)} observations are available."
        )

    recent_df = (
        historical_df
        .tail(months)
    )

    return historical_state_shares(
        recent_df
    )


# =========================================================
# Top-down allocation
# =========================================================

def allocate_aggregate_forecast(
    aggregate_forecast,
    shares,
):
    """
    Allocate aggregate forecasts to NY, CO, and TX.

    Parameters
    ----------
    aggregate_forecast : array-like
        Aggregate forecasts for future periods.

    shares : dict
        State allocation shares.

        Example:
        {
            "NY": 0.25,
            "CO": 0.20,
            "TX": 0.55
        }

    Returns
    -------
    pandas.DataFrame
        Columns:
            Aggregate
            NY
            CO
            TX
    """

    forecast = (
        _validate_aggregate_forecast(
            aggregate_forecast
        )
    )

    for state in VALID_STATES:

        if state not in shares:
            raise ValueError(
                f"Missing allocation share for {state}."
            )

    share_values = np.array(
        [
            shares[state]
            for state in VALID_STATES
        ],
        dtype=float,
    )

    if np.any(
        share_values < 0
    ):
        raise ValueError(
            "State shares cannot be negative."
        )

    if not np.isclose(
        share_values.sum(),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "State shares must sum to 1."
        )

    result = pd.DataFrame(
        {
            "Aggregate": forecast,
        }
    )

    for state in VALID_STATES:

        result[state] = (
            forecast
            * shares[state]
        )

    return result


def top_down_forecast(
    aggregate_forecast,
    historical_df,
    allocation_rule="recent_3_months",
):
    """
    Apply a top-down forecasting allocation rule.

    Supported rules
    ---------------
    recent_3_months
        Allocate based on state shares during the most
        recent three historical months.

    historical
        Allocate based on cumulative state shares across
        the full supplied historical dataset.

    Returns
    -------
    dict
        Contains:
            allocation_rule
            shares
            forecasts
    """

    if allocation_rule == "recent_3_months":

        shares = recent_state_shares(
            historical_df,
            months=3,
        )

    elif allocation_rule == "historical":

        shares = historical_state_shares(
            historical_df
        )

    else:

        raise ValueError(
            "allocation_rule must be either "
            "'recent_3_months' or 'historical'."
        )

    forecasts = (
        allocate_aggregate_forecast(
            aggregate_forecast,
            shares,
        )
    )

    return {
        "allocation_rule": allocation_rule,
        "shares": shares,
        "forecasts": forecasts,
    }


# =========================================================
# Bottom-up aggregation
# =========================================================

def combine_bottom_up_forecasts(
    ny_forecast,
    co_forecast,
    tx_forecast,
):
    """
    Combine independently generated state forecasts into
    an aggregate bottom-up forecast.

    Parameters
    ----------
    ny_forecast : array-like
    co_forecast : array-like
    tx_forecast : array-like

    Returns
    -------
    pandas.DataFrame
        Columns:
            Aggregate
            NY
            CO
            TX
    """

    ny = np.asarray(
        ny_forecast,
        dtype=float,
    )

    co = np.asarray(
        co_forecast,
        dtype=float,
    )

    tx = np.asarray(
        tx_forecast,
        dtype=float,
    )

    lengths = {
        len(ny),
        len(co),
        len(tx),
    }

    if len(lengths) != 1:
        raise ValueError(
            "NY, CO, and TX forecasts must have the same length."
        )

    if len(ny) == 0:
        raise ValueError(
            "State forecasts cannot be empty."
        )

    for state_name, forecast in [
        ("NY", ny),
        ("CO", co),
        ("TX", tx),
    ]:

        if np.isnan(
            forecast
        ).any():

            raise ValueError(
                f"{state_name} forecast contains missing values."
            )

        if np.any(
            forecast < 0
        ):

            raise ValueError(
                f"{state_name} forecast contains negative values."
            )

    aggregate = (
        ny
        + co
        + tx
    )

    return pd.DataFrame(
        {
            "Aggregate": aggregate,
            "NY": ny,
            "CO": co,
            "TX": tx,
        }
    )


# =========================================================
# Actual-data preparation
# =========================================================

def prepare_actual_hierarchy(
    actual_df,
):
    """
    Prepare actual aggregate and state data in the same
    structure used by hierarchical forecasts.

    Returns
    -------
    pandas.DataFrame
        Columns:
            Aggregate
            NY
            CO
            TX
    """

    _validate_states(
        actual_df
    )

    result = actual_df[
        VALID_STATES
    ].copy()

    result["Aggregate"] = (
        result["NY"]
        + result["CO"]
        + result["TX"]
    )

    return result[
        [
            "Aggregate",
            "NY",
            "CO",
            "TX",
        ]
    ].reset_index(
        drop=True
    )


# =========================================================
# Hierarchical forecast evaluation
# =========================================================

def evaluate_hierarchical_forecast(
    actual_df,
    forecast_df,
):
    """
    Evaluate forecast performance at the aggregate and
    state levels.

    Parameters
    ----------
    actual_df : pandas.DataFrame
        Must contain NY, CO, and TX.
        Aggregate is calculated internally.

    forecast_df : pandas.DataFrame
        Must contain:
            Aggregate
            NY
            CO
            TX

    Returns
    -------
    dict
        Example:

        {
            "Aggregate": {
                "MAPE": ...,
                ...
            },
            "NY": {...},
            "CO": {...},
            "TX": {...}
        }
    """

    actual = (
        prepare_actual_hierarchy(
            actual_df
        )
    )

    required_forecast_columns = [
        "Aggregate",
        "NY",
        "CO",
        "TX",
    ]

    missing = [
        col
        for col
        in required_forecast_columns
        if col
        not in forecast_df.columns
    ]

    if missing:
        raise ValueError(
            "Forecast data are missing columns: "
            + ", ".join(
                missing
            )
        )

    if len(
        actual
    ) != len(
        forecast_df
    ):

        raise ValueError(
            "Actual and forecast data must have "
            "the same number of periods."
        )

    results = {}

    for region in [
        "Aggregate",
        "NY",
        "CO",
        "TX",
    ]:

        results[
            region
        ] = calculate_metrics(
            actual[
                region
            ].to_numpy(),
            forecast_df[
                region
            ].to_numpy(),
        )

    return results


# =========================================================
# Comparison-table helper
# =========================================================

def create_hierarchy_metrics_table(
    metrics,
):
    """
    Convert hierarchical forecast metrics into a compact
    comparison table.

    Parameters
    ----------
    metrics : dict
        Output from evaluate_hierarchical_forecast().

    Returns
    -------
    pandas.DataFrame
    """

    rows = []

    for region in [
        "Aggregate",
        "NY",
        "CO",
        "TX",
    ]:

        region_metrics = (
            metrics[
                region
            ]
        )

        rows.append(
            {
                "Region": region,
                "MFE": region_metrics[
                    "MFE"
                ],
                "MAE": region_metrics[
                    "MAE"
                ],
                "MSE": region_metrics[
                    "MSE"
                ],
                "MAPE": region_metrics[
                    "MAPE"
                ],
                "Tracking Signal": (
                    region_metrics[
                        "Tracking Signal"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# Top-down vs bottom-up comparison
# =========================================================

def compare_hierarchical_methods(
    top_down_metrics,
    bottom_up_metrics,
):
    """
    Create a direct comparison of top-down and bottom-up
    MAPE by forecasting level.

    Returns
    -------
    pandas.DataFrame
    """

    rows = []

    for region in [
        "Aggregate",
        "NY",
        "CO",
        "TX",
    ]:

        rows.append(
            {
                "Region": region,
                "Top-Down MAPE": (
                    top_down_metrics[
                        region
                    ]["MAPE"]
                ),
                "Bottom-Up MAPE": (
                    bottom_up_metrics[
                        region
                    ]["MAPE"]
                ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    comparison[
        "Difference (Top-Down - Bottom-Up)"
    ] = (
        comparison[
            "Top-Down MAPE"
        ]
        - comparison[
            "Bottom-Up MAPE"
        ]
    )

    return comparison